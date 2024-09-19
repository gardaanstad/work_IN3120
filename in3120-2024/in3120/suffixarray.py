# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import sys
from bisect import bisect_left
from itertools import takewhile
from typing import Any, Dict, Iterator, Iterable, Tuple, List
from collections import Counter
from .corpus import Corpus
from .normalizer import Normalizer
from .tokenizer import Tokenizer



class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!

    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self.__corpus = corpus
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        self.__haystack: List[Tuple[int, str]] = []  # The (<document identifier>, <searchable content>) pairs.
        self.__suffixes: List[Tuple[int, int]] = []  # The sorted (<haystack index>, <start offset>) pairs.
        self.__build_suffix_array(fields)  # Construct the haystack and the suffix array itself.
    
    def __build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """
        
        documents = iter(self.__corpus) # [document, document, document]
        doc = next(documents, None) # document_id = 0, fields = {'a': 'This subject is', 'b': 'Great'}
        
        for doc in self.__corpus:
            content: str = " ".join(doc.get_field(field, "") for field in fields) # "This subject is Great"
            normalized: str = self.__normalize(content) # "this subject is great"
            
            self.__haystack.append((doc.get_document_id(), normalized)) # (0, "this subject is great")
                                                                        # (0, 0), (0, 5) ... (10, 0), (10, 3)
            
            doc = next(documents, None)

        # (haystack_index, string offset)
        # doc1:"abcd" -> (0, 0), (0, 1), (0, 2), ..., (0, 4)
        # doc2:"dcba" -> (1, 0), (1, 1), (1, 2), ..., (1, 4)
        
        suffixes = []
        for i, (_, content) in enumerate(self.__haystack):
            for offset in range(len(content)):
                is_start_of_token = (offset == 0 or content[offset - 1] == " ")
                
                if is_start_of_token:
                    suffixes.append((i, offset))
        
        # sorted suffixes: (1, 3), (0, 0), (1, 2), (0, 1), (1, 1), (0, 2), (0, 3), (1, 0)
        self.__suffixes = sorted(suffixes, 
                                 key=lambda index_offset: self.__haystack[index_offset[0]][1][index_offset[1]:])
        
        if "TEST" in fields:
            print("\n\nTEST\n")
            print(f"Haystack: {self.__haystack}")
            print(f"\nSuffixes: {[(haystack, offset) for haystack, offset in self.__suffixes]}")
            print()
            
    def __normalize(self, buffer: str) -> str:
        """
        Produces a normalized version of the given string. Both queries and documents need to be
        identically processed for lookups to succeed.
        """
        
        canonicalized = self.__normalizer.canonicalize(buffer)
        strings = self.__tokenizer.strings(canonicalized)
        normalized = [self.__normalizer.normalize(token) for token in strings]
        
        return " ".join(normalized)

    def __binary_search(self, needle: str) -> int:
        """
        Does a binary search for a given normalized query (the needle) in the suffix array (the haystack).
        Returns the position in the suffix array where the normalized query is either found, or, if not found,
        should have been inserted.

        Kind of silly to roll our own binary search instead of using the bisect module, but seems needed
        prior to Python 3.10 due to how we represent the suffixes via (index, offset) tuples. Version 3.10
        added support for specifying a key.
        """
        
        return bisect_left(self.__suffixes, needle, key=lambda x: self.__haystack[x[0]][1][x[1]:]) #haystack[id][searchable content][offset:]

    def evaluate(self, query: str, options: dict) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.

        The client can supply a dictionary of options that controls this query evaluation process: The maximum
        number of documents to return to the client is controlled via the "hit_count" (int) option.

        The results yielded back to the client are dictionaries having the keys "score" (int) and
        "document" (Document).
        """
        
        normalized_query = self.__normalize(query)
        if not normalized_query:
            return

        hit_count = options.get('hit_count', 5)
        matches = Counter()

        start_index = self.__binary_search(normalized_query)
        
        for haystack_index, start_offset in self.__suffixes[start_index:]:
            suffix = self.__haystack[haystack_index][1][start_offset:] # suffix = 'subject is great'
            
            if not suffix.startswith(normalized_query): # if the suffix does not start with the normalized query, break the loop
                break
            
            matches[self.__haystack[haystack_index][0]] += 1 # {doc_id: score (count of times the query appears in the doc)}

        for document_id, score in matches.most_common(hit_count):
            yield {"score": score, "document": self.__corpus.get_document(document_id)}