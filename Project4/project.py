# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    
    def editor(text):
        start_seq = r"\*{3} START\s(?:[A-Z]+\s)+PROJECT\s(?:[A-Z]+\s)+\*{3}"
        end_seq = r"\*{3} END\s(?:[A-Z]+\s)+PROJECT\s(?:[A-Z]+\s)+\*{3}"    
        return text[re.search(start_seq, text).span()[1]: re.search(end_seq, text).span()[0]]

    contents = requests.get(url).text
    contents = editor(contents)
    windows_nl = r"(\r\n)"
    contents = re.sub(windows_nl, "\n", contents)
    return contents


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    
    sol = re.sub(r"^(\s)+", "", book_string)
    sol = re.sub(r"(\n){2,}", "\x03\x02", sol)
    sol = re.sub(r"(\n{1})", " ", sol)
    token_seq = r"(\x02|\x03|\w+|[?'’,.;:()\-]|\")"
    token_list = re.findall(token_seq, sol)
    token_list.insert(0, "\x02")  
    token_list.append("\x03")
    
    return token_list


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        
        unique_words = pd.Series(tokens).nunique()
        token_freq = pd.Series(tokens).value_counts()
        sol = token_freq.apply(lambda x: 1 / unique_words)

        return sol
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        
        sol = 1.0
        for i in words:
            if i in self.mdl.index:
                sol *= self.mdl[i]
            else:
                return 0
        
        return sol
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        
        random_sample = np.random.choice(self.mdl.index, size = M, replace = True, p = self.mdl.values)
        return " ".join(random_sample)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        
        token_counts = len(tokens)
        token_freq = pd.Series(tokens).value_counts()
        sol = token_freq.apply(lambda x: x / token_counts)

        return sol
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        
        sol = 1.0
        for i in words:
            if i in self.mdl.index:
                sol *= self.mdl[i]
            else:
                return 0

        return sol
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        
        random_sample = np.random.choice(self.mdl.index, size = M, replace = True, p = self.mdl.values)
        return " ".join(random_sample)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            # self.prev_mdl = mdl.prev_mdl
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        
        sol = list()
        for i in range(1 + len(tokens) - self.N):
            temp = list()
            for j in range(self.N):
                temp.append(tokens[i + j])
                
            sol.append(tuple(temp))

        return sol
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
        # ngram counts C(w_1, ..., w_n)
        ...
        # n-1 gram counts C(w_1, ..., w_(n-1))
        ...

        # Create the conditional probabilities
        ...
        
        # Put it all together

        ng_col = pd.Series(ngrams)
        ng_freqs = ng_col.value_counts()
        n1_col = ng_col.apply(lambda x: str(x[:-1]))
        n1_freqs = n1_col.value_counts()

        df = pd.DataFrame(ng_freqs).reset_index().rename(columns = {0: "ngrams_counts", "index": "ngram"})
        df["n1gram"] = df["ngram"].apply(lambda x: x[:-1])
        df["n1gram_counts"] = df["n1gram"].apply(lambda x: n1_freqs[str(x)])
        df["prob"] = df["ngrams_counts"] / df["n1gram_counts"]

        return df.drop(columns = ["ngrams_counts", "n1gram_counts"])
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        
        ngrams = self.create_ngrams(words)
        sol = 1.0
        
        for i in ngrams:
            temp = self.mdl["ngram"] == i
            if temp.sum() == 0:
                return 0.0
            else:
                sol *= self.mdl["prob"][temp].iloc[0]

        sol *= self.prev_mdl.probability(words[: self.N - 1])

        return sol

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        # Use a helper function to generate sample tokens of length `length`
        ...
        
        # Transform the tokens to strings
        
        def single_sampler(self, sol):
                if len(sol) < self.N - 1:  # process words before reach N
                    self.prev_mdl.sample_helper(sol)

                unique = tuple(sol[-self.N + 1:])
                check = (self.mdl["n1gram"] == unique)

                if check.sum() != 0:
                    match_mdl = self.mdl[check].set_index("ngram")
                    random_token = np.random.choice(match_mdl.index, size = 1, p = match_mdl["prob"].values)
                    sol.append(random_token[0][-1])

                else:
                    sol.append("\x03")
                    return

        sol = ["\x02"]
        while len(sol) != M:
            single_sampler(self, sol)

        sol.insert(0, "\x02")
        return " ".join(sol).strip()
    
    
        