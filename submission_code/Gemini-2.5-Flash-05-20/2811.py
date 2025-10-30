class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        selected_numbers = set()  # Stores the distinct positive integers chosen for the array
        current_sum = 0           # Accumulates the sum of the chosen numbers
        num_count = 0             # Counts how many numbers have been selected
        candidate = 1             # The current positive integer being considered for selection

        # Loop until we have selected 'n' distinct positive integers
        while num_count < n:
            # Determine the 'complement' number that would sum to k with the current candidate.
            complement = k - candidate

            # Check if adding 'candidate' would violate the k-avoiding condition:
            # The condition 'candidate not in selected_numbers' is inherently handled by
            # how 'candidate' is incremented and `selected_numbers` is a set.
            # The crucial part is checking if 'k - candidate' (its complement) is already in the set.
            if complement not in selected_numbers:
                # If the complement is not in the set, 'candidate' can be safely added.
                selected_numbers.add(candidate)
                current_sum += candidate
                num_count += 1
            
            # Move to the next positive integer to consider
            candidate += 1
            
        return current_sum