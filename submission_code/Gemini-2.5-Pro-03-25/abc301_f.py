# YOUR CODE HERE
import sys
# Using defaultdict simplifies adding counts to potentially new states
from collections import defaultdict

def solve():
    # Read input string
    S = sys.stdin.readline().strip()
    N = len(S)
    MOD = 998244353

    # Precompute modular inverse of 26 modulo MOD
    # Since 26 = 2 * 13 and MOD = 998244353 is a prime larger than 13,
    # the modular inverse exists.
    # Fermat's Little Theorem: a^(p-2) === a^(-1) (mod p) for prime p
    inv26 = pow(26, MOD - 2, MOD)

    # Initialize DP state map. 
    # Key is a tuple (k1, k2, k3) representing the state.
    # Value is the count of ways to reach this state configuration.
    # k1: number of distinct uppercase letters X for which the longest prefix of XXaX formed is X.
    # k2: number of distinct uppercase letters X for which the longest prefix of XXaX formed is XX.
    # k3: number of distinct uppercase letters X for which the longest prefix of XXaX formed is XXa.
    # k0 (implicit): number of distinct uppercase letters X for which the longest prefix is empty string. k0 = 26 - k1 - k2 - k3.
    
    # Using defaultdict avoids checking if a state exists before adding to it.
    dp = defaultdict(int) 
    # Initial state: empty prefix, no progress towards any XXaX. All 26 letters are state 0.
    # So k1=0, k2=0, k3=0. There is 1 way to be in this state (empty prefix).
    dp[(0, 0, 0)] = 1

    # Iterate through the input string S character by character
    for i in range(N):
        # Prepare a new DP state map for the next step (i+1)
        new_dp = defaultdict(int)
        
        # If the current dp map is empty, it means there are no valid paths left.
        # We can break early as all subsequent states will also have 0 count.
        if not dp:
            break

        # Get the current character from S
        char = S[i]

        # Iterate through all states reached at step i and their counts
        for state, count in dp.items():
            
            # Extract state components (k1, k2, k3)
            k1, k2, k3 = state
            
            # Calculate k0, the number of letters in state 0
            current_sum = k1 + k2 + k3
            # Basic sanity check: if state components sum > 26 or any are negative, it's an invalid state.
            # This shouldn't happen with correct logic, but acts as a safeguard.
            if current_sum > 26 or k1 < 0 or k2 < 0 or k3 < 0:
                 continue 
            
            k0 = 26 - current_sum

            # Handle transitions based on the current character S[i]
            
            # Case 1: S[i] is '?'
            if char == '?':
                # '?' can be replaced by any of 52 letters. We consider uppercase and lowercase replacements separately.

                # Subcase: replace '?' with an uppercase letter Y (26 possibilities)
                # The effect depends on the state of the chosen letter Y.
                # We use the aggregated counts k0, k1, k2, k3 to determine transitions.
                
                # If Y was in state 0: it moves to state 1. This applies to k0 letters.
                # New state: (k1+1, k2, k3). Total contribution: count * k0.
                if k0 > 0:
                    new_state = (k1 + 1, k2, k3)
                    # Ensure new state is valid (sum <= 26)
                    if current_sum + 1 <= 26:
                         new_dp[new_state] = (new_dp[new_state] + count * k0) % MOD
                
                # If Y was in state 1: it moves to state 2. This applies to k1 letters.
                # New state: (k1-1, k2+1, k3). Total contribution: count * k1.
                if k1 > 0:
                    new_state = (k1 - 1, k2 + 1, k3)
                    # New state sum is k1-1+k2+1+k3 = current_sum. Always valid if current was.
                    new_dp[new_state] = (new_dp[new_state] + count * k1) % MOD

                # If Y was in state 2: it stays in state 2. This applies to k2 letters.
                # New state: (k1, k2, k3). Total contribution: count * k2.
                if k2 > 0:
                    new_state = (k1, k2, k3) 
                    # State sum is current_sum. Valid.
                    new_dp[new_state] = (new_dp[new_state] + count * k2) % MOD
                
                # If Y was in state 3: it moves to state 4 (invalid state). This applies to k3 letters.
                # The corresponding count * k3 ways lead to strings containing XXaX and are discarded.

                # Subcase: replace '?' with a lowercase letter a (26 possibilities)
                # Seeing a lowercase letter 'a' potentially advances progress for all X that have seen XX.
                # Any letter X currently in state 2 (XX seen) moves to state 3 (XXa seen).
                # Any letter X currently in state 3 (XXa seen) remains in state 3.
                # Letters in states 0 and 1 are unaffected regarding their state number.
                # New state configuration counts: k1' = k1, k2' = 0, k3' = k2 + k3.
                new_state = (k1, 0, k2 + k3)
                # New state sum is k1 + 0 + k2 + k3 = current_sum. Valid.
                # Total contribution: count * 26 (for 26 choices of lowercase letter).
                new_dp[new_state] = (new_dp[new_state] + count * 26) % MOD

            # Case 2: S[i] is a fixed lowercase letter
            elif 'a' <= char <= 'z':
                # Similar transition as replacing '?' with lowercase, but only 1 way (specific character).
                new_state = (k1, 0, k2 + k3)
                # New state sum is current_sum. Valid.
                # Total contribution: count * 1.
                new_dp[new_state] = (new_dp[new_state] + count) % MOD

            # Case 3: S[i] is a fixed uppercase letter Y
            elif 'A' <= char <= 'Z':
                # The transition depends on the state of the specific letter Y before this step.
                # Since the aggregated state (k1, k2, k3) doesn't store this information,
                # we use a logic based on symmetry assumption and modular inverse.
                # It assumes that for the 'count' ways to reach state (k1, k2, k3),
                # the specific letter Y is in state p (0, 1, 2, or 3) in a fraction k_p / 26 of these ways.
                
                # If Y was in state 0: moves 0 -> 1. New state (k1+1, k2, k3).
                # Contribution: count * (k0 / 26) = count * k0 * inv26 (mod MOD).
                if k0 > 0:
                    new_state = (k1 + 1, k2, k3)
                    if current_sum + 1 <= 26:
                       term = (count * k0 * inv26) % MOD
                       new_dp[new_state] = (new_dp[new_state] + term) % MOD
                
                # If Y was in state 1: moves 1 -> 2. New state (k1-1, k2+1, k3).
                # Contribution: count * (k1 / 26) = count * k1 * inv26 (mod MOD).
                if k1 > 0:
                    new_state = (k1 - 1, k2 + 1, k3)
                    # State sum current_sum. Valid.
                    term = (count * k1 * inv26) % MOD
                    new_dp[new_state] = (new_dp[new_state] + term) % MOD

                # If Y was in state 2: moves 2 -> 2. New state (k1, k2, k3).
                # Contribution: count * (k2 / 26) = count * k2 * inv26 (mod MOD).
                if k2 > 0:
                    new_state = (k1, k2, k3) 
                    # State sum current_sum. Valid.
                    term = (count * k2 * inv26) % MOD
                    new_dp[new_state] = (new_dp[new_state] + term) % MOD

                # If Y was in state 3: moves 3 -> 4 (invalid).
                # Contribution to invalid paths: count * (k3 / 26) = count * k3 * inv26 (mod MOD).
                # These paths are discarded.
        
        # Update the DP state map for the next iteration
        dp = new_dp

    # After processing the entire string, the total number of valid ways
    # is the sum of counts over all final states in the dp map.
    total_valid_ways = 0
    for count in dp.values():
        total_valid_ways = (total_valid_ways + count) % MOD

    # Print the final answer
    print(total_valid_ways)

# Execute the solve function
solve()