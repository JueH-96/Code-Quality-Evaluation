def solve():
    MOD = 998244353
    
    N = int(input())
    A = list(map(int, input().split()))
    
    # dp[i] = number of Polish sequences of length i
    dp = [0] * (N + 1)
    dp[0] = 0  # empty sequence doesn't count
    dp[1] = 1  # only (0) is Polish of length 1
    
    # For each length
    for length in range(2, N + 1):
        # Try each possible first element
        for first in range(length):
            if first == 0:
                # (0) followed by nothing - but this gives length 1, not length
                continue
            
            # We need to partition the remaining length-1 elements into 'first' Polish sequences
            # Use dp2[i][j] = number of ways to form i Polish sequences using j elements
            dp2 = [[0] * length for _ in range(first + 1)]
            dp2[0][0] = 1
            
            for i in range(first):
                for j in range(length):
                    if dp2[i][j] == 0:
                        continue
                    # Add a Polish sequence of length k
                    for k in range(1, length - j):
                        if j + k < length and dp[k] > 0:
                            dp2[i + 1][j + k] = (dp2[i + 1][j + k] + dp2[i][j] * dp[k]) % MOD
            
            if dp2[first][length - 1] > 0:
                dp[length] = (dp[length] + dp2[first][length - 1]) % MOD
    
    # Now count Polish sequences of length N that are ≤ A lexicographically
    def count_polish_leq(pos, target, is_limit):
        if pos == N:
            return 1 if is_polish_sequence(target) else 0
        
        result = 0
        max_val = A[pos] if is_limit else N - 1
        
        for digit in range(max_val + 1):
            new_target = target + [digit]
            new_limit = is_limit and (digit == A[pos])
            result = (result + count_polish_leq(pos + 1, new_target, new_limit)) % MOD
        
        return result
    
    def is_polish_sequence(seq):
        if not seq:
            return False
        if len(seq) == 1:
            return seq[0] == 0
        
        first = seq[0]
        if first == 0 or first >= len(seq):
            return False
        
        # Check if we can partition the rest into 'first' Polish sequences
        return can_partition_into_polish(seq[1:], first)
    
    def can_partition_into_polish(seq, num_parts):
        if num_parts == 0:
            return len(seq) == 0
        if not seq:
            return False
        
        # Try all possible lengths for the first Polish sequence
        for length in range(1, len(seq) + 1):
            if is_polish_sequence(seq[:length]):
                if can_partition_into_polish(seq[length:], num_parts - 1):
                    return True
        return False
    
    # More efficient approach using memoization
    memo = {}
    
    def count_sequences(pos, is_limit):
        if pos == N:
            return 0
        
        if not is_limit and pos in memo:
            return memo[pos]
        
        result = 0
        max_val = A[pos] if is_limit else N - 1
        
        # If we're starting a new sequence
        if pos == 0:
            for first in range(max_val + 1):
                if first == 0 and N == 1:
                    result = (result + 1) % MOD
                elif first > 0:
                    # Count ways to place 'first' Polish sequences in remaining positions
                    result = (result + count_with_first(first, 1, pos + 1, is_limit and first == A[0])) % MOD
        
        if not is_limit:
            memo[pos] = result
        return result
    
    def count_with_first(needed, used, pos, is_limit):
        if needed == 0:
            return 1 if used == N else 0
        if used >= N:
            return 0
        
        result = 0
        # Try Polish sequences of different lengths
        for seq_len in range(1, N - used + 1):
            if used + seq_len <= N:
                ways = count_polish_of_length_leq(seq_len, used, is_limit)
                if ways > 0:
                    result = (result + ways * count_with_first(needed - 1, used + seq_len, pos + seq_len, is_limit)) % MOD
        
        return result
    
    def count_polish_of_length_leq(length, start_pos, is_limit):
        if length == 1:
            if is_limit and start_pos < N and A[start_pos] >= 0:
                return 1
            elif not is_limit:
                return 1
            else:
                return 0
        
        result = 0
        max_first = N - 1
        if is_limit and start_pos < N:
            max_first = min(max_first, A[start_pos])
        
        for first in range(min(length, max_first + 1)):
            if first == 0:
                continue
            
            # Count ways to have 'first' Polish sequences in remaining length-1 positions
            ways = count_partitions(first, length - 1, start_pos + 1, is_limit and start_pos < N and first == A[start_pos])
            result = (result + ways) % MOD
        
        return result
    
    def count_partitions(num_seqs, total_len, start_pos, is_limit):
        if num_seqs == 0:
            return 1 if total_len == 0 else 0
        if total_len <= 0:
            return 0
        
        result = 0
        for seq_len in range(1, total_len + 1):
            ways = count_polish_of_length_leq(seq_len, start_pos, is_limit)
            if ways > 0:
                result = (result + ways * count_partitions(num_seqs - 1, total_len - seq_len, start_pos + seq_len, is_limit)) % MOD
        
        return result
    
    print(count_sequences(0, True))

solve()