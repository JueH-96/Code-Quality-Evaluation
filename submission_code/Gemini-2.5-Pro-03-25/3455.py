import collections

class Node:
    """
    Represents a character node in the string. 
    It stores the character and pointers to the previous and next nodes 
    *with the same character* in the original string order.
    This structure allows efficient identification and update of character sequences.
    """
    def __init__(self, char):
        self.char = char
        # Pointer to the previous node with the same character.
        self.prev_char = None
        # Pointer to the next node with the same character.
        self.next_char = None
        # Flag indicating if this node has been effectively deleted by an operation.
        self.deleted = False
    
    # Node objects are hashable by their default object identity (memory address),
    # which is sufficient for use as keys in sets or dictionaries when tracking triplets.

class Solution:
    """
    Solves the minimum length problem by simulating the reduction process.
    It uses a doubly linked list structure for each character type to efficiently 
    find and process removable triplets of characters.
    """
    def minimumLength(self, s: str) -> int:
        """
        Calculates the minimum possible length of the string `s` after applying 
        the specified reduction process any number of times.

        The process involves finding a character `s[i]` that has identical characters
        `s[j]` (closest to its left) and `s[k]` (closest to its right), and then
        deleting `s[j]` and `s[k]`. This is equivalent to finding three consecutive 
        occurrences `(j, i, k)` of the same character and removing the outer two (`j` and `k`).

        Args:
            s: The input string consisting of lowercase English letters.

        Returns:
            The minimum length of the final string achievable.
        """
        n = len(s)
        # If the string has 0 or 1 characters, no operations can be performed.
        if n <= 1:
            return n

        # Create Node objects for each character. We don't need to store the original index
        # within the Node itself after the initial setup.
        nodes = [Node(s[i]) for i in range(n)] 
        
        # Build the character-specific linked lists.
        # For each character 'a' through 'z', this connects all nodes containing that character
        # in the order they appear in the string, using prev_char and next_char pointers.
        char_last_node = {} # Tracks the most recently encountered node for each character.
        for i in range(n):
            node = nodes[i]
            char = node.char
            # If we've seen this character before, link the current node to the previous one.
            if char in char_last_node:
                prev_node = char_last_node[char]
                node.prev_char = prev_node
                prev_node.next_char = node
            # Update the last seen node for this character.
            char_last_node[char] = node 

        # Initialize a queue (deque for efficiency) to store potential triplets (nj, ni, nk) 
        # that represent possible removal operations. nj and nk would be removed, ni is the pivot.
        q = collections.deque()
        # Initialize a set to keep track of which triplets are currently considered valid 
        # candidates for removal. Using a set provides efficient lookups and removals.
        valid_triplets = set() 

        # Find all initial triplets present in the string.
        # A node `ni` can act as a pivot if it has both a `prev_char` (nj) and `next_char` (nk).
        for i in range(n):
            ni = nodes[i]
            nj = ni.prev_char
            nk = ni.next_char
            
            if nj is not None and nk is not None:
                # Form the triplet tuple using the node objects themselves.
                triplet = (nj, ni, nk)
                # Add the newly found triplet to the queue and the set of valid triplets.
                # The check `if triplet not in valid_triplets` prevents duplicates if structure allows, though unlikely here.
                if triplet not in valid_triplets: 
                     q.append(triplet)
                     valid_triplets.add(triplet)
                
        deleted_count = 0 # Keep track of the total number of nodes marked as deleted.

        # Process the queue as long as there are potential operations.
        while q:
            # Get the next potential triplet operation from the queue.
            triplet = q.popleft()
            nj, ni, nk = triplet # Unpack the nodes from the triplet tuple.

            # --- Validity Checks --- Perform checks before processing.
            
            # 1. Check if this specific triplet is still in the set of valid triplets.
            #    It might have been invalidated and removed from the set by a prior operation.
            if triplet not in valid_triplets: 
                 continue # Skip if no longer considered valid.

            # 2. Check if any of the nodes involved in this triplet have already been deleted
            #    by a *different* operation. If so, this triplet cannot be processed.
            if nj.deleted or ni.deleted or nk.deleted:
                 # If the triplet involves deleted nodes, it's inherently invalid.
                 # Remove it from the valid set if it's still present (cleanup).
                 if triplet in valid_triplets:
                     valid_triplets.remove(triplet)
                 continue # Skip this invalid triplet.

            # --- Process the Valid Triplet ---
            # Mark the left node (nj) and right node (nk) associated with this operation as deleted.
            nj.deleted = True
            nk.deleted = True
            deleted_count += 2 # Increment the count of deleted nodes.
            
            # Remove this triplet from the set of valid triplets because it has now been processed.
            valid_triplets.remove(triplet) 

            # Identify the neighbors of nj and nk in the character list *before* modifying pointers.
            # pj is the node before nj (if any). sk is the node after nk (if any).
            pj = nj.prev_char 
            sk = nk.next_char

            # Update the character-specific linked list pointers to bypass the deleted nodes nj and nk.
            # This effectively removes nj and nk from the sequence of nodes for their character.
            
            # Connect Pj <-> Ni (skipping Nj)
            if pj: # If pj exists (nj was not the first node for this character)
                pj.next_char = ni # pj's next node becomes ni
            ni.prev_char = pj # ni's previous node becomes pj
            
            # Connect Ni <-> Sk (skipping Nk)
            ni.next_char = sk # ni's next node becomes sk
            if sk: # If sk exists (nk was not the last node for this character)
                sk.prev_char = ni # sk's previous node becomes ni

            # --- Check for New Triplets ---
            # The removal of nj and nk might have created new consecutive triplets involving
            # the neighboring nodes pj, ni, and sk. We need to check these possibilities.
            
            # Check around Pj: Can Pj now form a triplet (Pj's prev, Pj, Ni)?
            if pj and not pj.deleted: # Check if Pj exists and is not deleted
                 ppj = pj.prev_char # Pj's own previous character neighbor
                 if ppj and not ppj.deleted: # Check if ppj exists and is not deleted
                     new_trip_pj = (ppj, pj, ni) # Form the potential new triplet
                     # If this potential triplet isn't already known or processed, add it.
                     if new_trip_pj not in valid_triplets:
                         valid_triplets.add(new_trip_pj)
                         q.append(new_trip_pj) # Add to the queue for later processing

            # Check around Ni: Can Ni now form a triplet (Pj, Ni, Sk)?
            # Ni itself was the pivot and remains undeleted. Check if its new neighbors Pj and Sk form a triplet.
            if not ni.deleted: # This condition is always true here since ni is the pivot.
                if pj and not pj.deleted and sk and not sk.deleted: # Both neighbors must exist and be undeleted.
                     new_trip_ni = (pj, ni, sk) # Form the potential new triplet
                     if new_trip_ni not in valid_triplets:
                         valid_triplets.add(new_trip_ni)
                         q.append(new_trip_ni)

            # Check around Sk: Can Sk now form a triplet (Ni, Sk, Sk's next)?
            if sk and not sk.deleted: # Check if Sk exists and is not deleted
                ssk = sk.next_char # Sk's own next character neighbor
                if ssk and not ssk.deleted: # Check if ssk exists and is not deleted
                    new_trip_sk = (ni, sk, ssk) # Form the potential new triplet
                    if new_trip_sk not in valid_triplets:
                         valid_triplets.add(new_trip_sk)
                         q.append(new_trip_sk)

        # After the queue is empty, no more operations can be performed.
        # The minimum length is the original length minus the total number of deleted nodes.
        return n - deleted_count