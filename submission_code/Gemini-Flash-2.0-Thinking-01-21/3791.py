from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        # DSU parent array. parent[i] stores the index of the first available basket
        # at or after original index i.
        # Initialize parent[i] = i, meaning initially index i is available and points to itself (is a root).
        # parent[n] = n acts as a sentinel indicating no more baskets.
        parent = list(range(n + 1))

        def find(i):
            """
            Finds the representative of the set containing index i, which in this DSU
            represents the index of the first available basket at or after original index i.
            Uses path compression.
            """
            if parent[i] == i:
                return i
            # Path compression: Make the parent of i the root itself.
            parent[i] = find(parent[i])
            return parent[i]

        unplaced_count = 0

        for fruit_quantity in fruits:
            # Start searching for a basket for the current fruit.
            # The rule is to find the *leftmost* available basket that fits.
            # The search conceptually starts from index 0 of the *original* basket list,
            # but we only check *available* baskets.
            
            # Find the index of the first available basket starting from index 0.
            # The find(0) call correctly returns the index of the first available
            # basket starting from index 0 by traversing the DSU structure.
            j = find(0)

            # Iterate through available baskets (by using find(j+1) to jump)
            # until we find one that fits or exhaust all baskets.
            
            found_basket = False

            while j < n:
                if baskets[j] >= fruit_quantity:
                    # Found a suitable basket at index j. Place the fruit.
                    
                    # Mark basket j as used. The next available basket at or after index j
                    # is now the same as the next available basket at or after index j + 1.
                    # We update the parent of j's set (which is j itself initially if available)
                    # to point to the representative of the set containing j+1.
                    # This effectively removes j from the chain of available baskets starting at or before j.
                    parent[j] = find(j + 1) 
                    
                    found_basket = True
                    # Fruit placed successfully. Break the inner search loop for this fruit.
                    break
                else:
                    # Basket j is available but too small for this fruit.
                    # Continue searching for a basket for THIS fruit from the next available basket
                    # strictly after index j.
                    # The next available index after j is found by find(j + 1).
                    j = find(j + 1)

            # If the inner while loop completed and found_basket is still False,
            # it means no suitable available basket was found for the current fruit.
            if not found_basket:
                unplaced_count += 1

        return unplaced_count