class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        """
        We have:
          - x pieces of "AA"
          - y pieces of "BB"
          - z pieces of "AB"
        
        We want to concatenate SOME (possibly all/none) of these pieces in any order
        so that the final string does not contain "AAA" or "BBB" as a substring.
        Return the maximum possible length of such a string.
        
        Key observations/constraints (each piece has length 2):
          1) We can never place two "AA" pieces back-to-back ("AAAA" causes 'AAA')
          2) We can never place two "BB" pieces back-to-back ("BBBB" causes 'BBB')
          3) We must check carefully how "AB" can be inserted without creating triples.
             - "AA"+"AB" would form "A A A B" => triple 'A', not allowed
             - "AB"+"AA" would form "A B A A" => OK
             - "BB"+"AB" would form "B B A B" => OK
             - "AB"+"BB" would form "A B B B" => triple 'B', not allowed
             - "AB"+"AB" is fine
             - "AA"+"BB" or "BB"+"AA" are also fine (no triples)
        
        Strategy for maximal length:
        
        1) Try to alternate all possible "AA" and "BB". Each valid alternation uses one "AA"
           and one "BB". Let n = min(x, y). Then we can use n "AA" and n "BB".
           - That yields 2n pieces, each of length 2 => total 4n characters so far.
           
           We can arrange them in two main ways (since we can reorder freely):
             A) A-chain: AA -> BB -> AA -> BB -> ... (start AA, end BB if n>0)
             B) B-chain: BB -> AA -> BB -> AA -> ... (start BB, end AA if n>0)
           
           Depending on how we arrange, the count of "BB"->"AA" adjacencies differs, 
           and those are exactly the spots where we can insert "AB" (since "BB"+"AB"+"AA"
           is valid, but "AA"+"AB"+"BB" is not).  

           - In the B-chain (start BB, then AA, etc.), if we have n pairs, we get n 
             “BB -> AA” adjacencies.
           - In the A-chain (start AA, then BB, etc.), if we have n pairs, we get n-1
             “BB -> AA” adjacencies (when n>0).

           For each "BB"->"AA" adjacency, we can optionally insert one "AB". 
           Each "AB" is length=2, so we gain +2 for each inserted "AB". 
           We can insert up to min(z, (number_of_BB_to_AA_adjacencies)).

        2) After using n "AA" and n "BB", there may be leftovers:
           leftover_x = x - n  (extra "AA"s)
           leftover_y = y - n  (extra "BB"s)
           
           - We cannot put two leftover "AA" or "BB" in a row (that would cause triples).
             But we can place at most one leftover "BB" in a position where the adjacent piece
             ends/starts with 'B'? Actually simpler is to note we can put at most one leftover
             piece at the front and at most one leftover piece at the end, if it does not cause
             triples.
             
           For the B-chain (start=BB, end=AA if n>0):
             - We can prepend "AA" (leftover_x>0) at the front, because "AA"+"BB" is valid.
               But not "BB"+"BB".
             - We can append "BB" (leftover_y>0) at the end, because "AA"+"BB" is valid.
               So we gain up to 2 leftover pieces => +2 length each => up to +4 total.
               
           For the A-chain (start=AA, end=BB if n>0):
             - We can prepend "BB" (leftover_y>0) at the front, because "BB"+"AA" is valid.
               But not "AA"+"AA".
             - We can append "AA" (leftover_x>0) at the end, because "BB"+"AA" is valid.
               So likewise, up to +4 total from leftovers.
        
        3) Another possibility: we may choose not to use "AA" or "BB" at all and only use
           some arrangement of the z "AB" pieces. Repeated "AB" => "ABABAB..." never creates
           triple letters. That yields 2*z length if we use all z "AB".
           We can possibly attach one "BB" at the front (if any available) and one "AA"
           at the end (if any available), since:
             - "BB" + "AB" => "B B A B" is OK; 
               adding more "BB" at the front would create triple-B, not allowed.
             - The chain of z "AB" ends in 'B', so "B"+"AA" => "BAA" is OK, 
               more "AA" would create triple-A, not allowed.
           So if we skip using any "AA"/"BB" in the chain, we can still add up to one leftover 
           "BB" in front and up to one leftover "AA" at the end. That can add up to +4 if both
           leftovers exist, or +2 if only one leftover type is non-zero.
        
        We compute the best length among:
          - A-chain arrangement
          - B-chain arrangement
          - Just the "AB" chain (possibly with one leftover "BB" in front and one leftover "AA" in back)
        
        This covers all optimal ways of mixing them without creating triple letters.
        """

        # 1) Pair up AA & BB
        n = min(x, y)
        leftover_x = x - n  # leftover "AA"
        leftover_y = y - n  # leftover "BB"

        # A-chain: start AA => BB => AA => BB => ...
        # Number of "BB->AA" transitions = max(0, n-1) if n>0
        transitionsA = n - 1 if n > 0 else 0
        baseA = 4 * n  # using n "AA" + n "BB", each piece length 2 => total 2*n pieces => 2*n*2=4n
        insertA = 2 * min(z, transitionsA)  # can insert "AB" in each "BB->AA" adjacency
        # leftover: can prepend "BB" if leftover_y>0, append "AA" if leftover_x>0
        leftoverA = 2 * (min(1, leftover_y) + min(1, leftover_x))
        chainA_len = baseA + insertA + leftoverA

        # B-chain: start BB => AA => BB => AA => ...
        # Number of "BB->AA" transitions = n (if n>0)
        transitionsB = n  # if n=0, that effectively becomes 0 anyway
        baseB = 4 * n
        insertB = 2 * min(z, transitionsB)
        # leftover: can prepend "AA" if leftover_x>0, append "BB" if leftover_y>0
        leftoverB = 2 * (min(1, leftover_x) + min(1, leftover_y))
        chainB_len = baseB + insertB + leftoverB

        # C-chain: just use up to z "AB"
        # We can add at most 1 "BB" in front and 1 "AA" in back (if available)
        # leftover_x = x, leftover_y = y if we are not using any "AA"/"BB" in the chain
        # because we skip them entirely in this scenario
        chainC_base = 2 * z  # use all z "AB"
        # Prepend "BB"? Only if y>0
        # Append "AA"? Only if x>0
        leftoverC = 2 * ((1 if y > 0 else 0) + (1 if x > 0 else 0))
        chainC_len = chainC_base + leftoverC

        return max(chainA_len, chainB_len, chainC_len)