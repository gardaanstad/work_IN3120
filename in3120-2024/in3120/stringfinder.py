# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

from typing import Iterator, Dict, Any, List, Tuple, Optional
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications and some minor
    NLP extensions. The running time of this algorithm is virtually independent of the size of the dictionary,
    and linear in the length of the buffer we are searching in.

    The tokenizer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    def __init__(self, trie: Trie, normalizer: Normalizer, tokenizer: Tokenizer):
        self.__trie = trie
        self.__normalizer = normalizer  # The same as was used for trie building.
        self.__tokenizer = tokenizer  # The same as was used for trie building.

    # Tokenize the input buffer
    # tokens = list(self.__tokenizer.strings(buffer))

    # Initialize the list of live states
    # Keep a list of "live states". As you iterate once over the tokens in the buffer, 
    # update and main this list of "live states". List comprehensions are your friend. :)
    #
    # The set of currently explored states. We represent a state as a triple consisting of
    # (a) a node in the trie that represents where in the trie we are after having consumed zero or more characters
    # (b) an index that represents the position into the original buffer where the state was "born"
    # (c) a string that represents the symbols consumed so far to get to the current state
    # 
    # (a) is what we advance along the way
    # (b) is needed so that we know where we first started if/when a match is found
    # (c) is needed so that we can differentiate between the surface
    #     form of the match and the (possibly heavily normalized) base form of the match.
    # live_states: List[Tuple[Trie, int, str]] = [(self.__trie, 0, "")]
    
    def scan(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Scans the given buffer and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        The matches, if any, are yielded back to the client as dictionaries having the keys "match" (str),
        "surface" (str), "meta" (Optional[Any]), and "span" (Tuple[int, int]). Note that "match" refers to
        the matching dictionary entry, "surface" refers to the content of the input buffer that triggered the
        match (the surface form), and "span" refers to the exact location in the input buffer where the surface
        form is found. Depending on the normalizer that is used, "match" and "surface" may or may not differ.

        A space-normalized version of the surface form is emitted as "surface", for convenience. Clients
        that require an exact surface form that is not space-normalized can easily reconstruct the desired
        string using the emitted "span" value.

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and more.
        """
        
        def __consume(state: Trie, token: str, consumed: str) -> Tuple[Optional[Trie], str]:
            """
            Internal helper method to reduce complexity in main function logic.
            Returns a tuple of [next Trie state after consuming token, or None if it doesn't exist],
            and [the correctly constructed consumed string]
            """
            next_state = state.consume(token)
            
            if next_state is not None:
                return next_state, consumed + token
            
            return state.consume(" " + token), consumed + " " + token
        
        # [(token, (start, end)), (token, (start, end)), ...]
        tokens = self.__tokenizer.tokens(buffer)        

        # initialize live_states on root Trie node, 0 start index and empty consumed
        live_states = [(self.__trie, 0, "")]
        
        for token, (start, end) in tokens:
            token = self.__normalizer.normalize(token)
            
            new_live_states = []
            states_to_remove = set()
            
            # iterate through live_states in reverse order, to make sure we prune/yield latest states first
            for state_index, (state, state_start, state_consumed) in reversed(list(enumerate(live_states))):
                
                # __consume() returns a Tuple of next_state Trie (or None), and the string of consumed
                next_state, next_consumed = __consume(state, token, state_consumed)
                
                # if branching from the current state with the current token is impossible, 
                # remove this irrelevant state from new_live_states (irrelevant state pruning)
                # if we're at the root state, don't remove it, just skip to next state (continue)
                if next_state is None:
                    if state_index == 0: # if at root
                        continue
                    
                    states_to_remove.add(state_index) # add irrelevant state's index to states_to_remove
                    continue
                
                # if first token in match, set state_start as the start of the current token
                if state_consumed == "":
                    state_start = start
                
                # found match (current state = ["token"], next state = [""])
                if next_state.is_final():
                    surface = " ".join(buffer[state_start:end].split()) # space normalized surface
                    span = (state_start, end) # start of current live state, end of current token
                    match = next_consumed
                    meta = next_state.get_meta()
                    
                    yield {
                        "surface": surface,
                        "span": span,
                        "match": match,
                        "meta": meta
                    }
                
                # don't append the state if it has no transitions (state is dead/already consumed + yielded)
                if len(next_state.transitions()) == 0:
                    continue
                
                new_live_states.append((next_state, state_start, next_consumed))
            
            # live_states = live_states excluding states_to_remove plus new_live_states
            live_states = [state for i, state in enumerate(live_states) if i not in states_to_remove] + new_live_states 