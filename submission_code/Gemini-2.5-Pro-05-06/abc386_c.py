import sys

def solve():
    # Read K. For this specific sub-problem, K is guaranteed to be 1.
    # We don't strictly need to use K in the logic if it's fixed,
    # but reading it is part of the input format.
    _ = int(sys.stdin.readline()) # K, not used explicitly as it's 1

    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    N = len(S)
    M = len(T)

    # Case 0: 0 operations needed
    # If S is already identical to T, 0 operations are performed.
    # This is within the "at most K=1 operations" limit.
    if S == T:
        print("Yes")
        return

    # If S != T, we must use exactly 1 operation.
    # (Since K=1, we can't use more than 1 operation).

    # Case 1: One replace operation
    # This is possible if S and T have the same length and differ by exactly one character.
    if N == M:
        diff_count = 0
        for i in range(N):
            if S[i] != T[i]:
                diff_count += 1
            # Optimization: if we've already found more than one difference,
            # it's not possible with a single replacement.
            if diff_count > 1:
                break
        
        if diff_count == 1:
            print("Yes")
            return
    
    # Case 2: One insert operation (to transform S into T)
    # This means T must be longer than S by exactly one character (M = N + 1).
    elif M == N + 1:
        # Find the first position `idx` where S and T differ, or `idx` becomes N if S is a prefix of T.
        # S = common_prefix + s_suffix
        # T = common_prefix + inserted_char + t_suffix
        # We need s_suffix to be equal to t_suffix.
        idx = 0
        while idx < N and S[idx] == T[idx]: # N is len(S)
            idx += 1
        
        # After the loop, S[:idx] == T[:idx].
        # The character T[idx] is potentially the one inserted.
        # So, the remainder of S (S[idx:]) must match the remainder of T after this potential insertion (T[idx+1:]).
        # This check correctly handles:
        #   - Insertion at the beginning (idx=0): S should be equal to T[1:].
        #   - Insertion in the middle (0 < idx < N): S[:idx]==T[:idx] and S[idx:]==T[idx+1:].
        #   - Insertion at the end (idx=N, S is a prefix of T): S should be equal to T[:-1].
        #     (Here S[N:] is "" and T[N+1:] which is T[M:] is also "")
        if S[idx:] == T[idx+1:]:
            print("Yes")
            return
            
    # Case 3: One delete operation (from S to transform S into T)
    # This means S must be longer than T by exactly one character (N = M + 1).
    elif N == M + 1:
        # Find the first position `idx` where S and T differ, or `idx` becomes M if T is a prefix of S.
        # S = common_prefix + char_to_delete + s_suffix
        # T = common_prefix + t_suffix
        # We need s_suffix to be equal to t_suffix.
        idx = 0
        while idx < M and S[idx] == T[idx]: # M is len(T)
            idx += 1
            
        # After the loop, S[:idx] == T[:idx].
        # The character S[idx] is potentially the one deleted.
        # So, the remainder of S after this potential deletion (S[idx+1:]) must match the remainder of T (T[idx:]).
        # This check correctly handles:
        #   - Deletion from the beginning (idx=0): S[1:] should be equal to T.
        #   - Deletion from the middle (0 < idx < M): S[:idx]==T[:idx] and S[idx+1:]==T[idx:].
        #   - Deletion from the end (idx=M, T is a prefix of S): S[:-1] should be equal to T.
        #     (Here S[M+1:] which is S[N:] is "" and T[M:] is also "")
        if S[idx+1:] == T[idx:]:
            print("Yes")
            return

    # If S != T and none of the 1-operation conditions were met,
    # or if len(S) and len(T) differ by more than 1 (abs(N-M) > 1),
    # then it's not possible with at most 1 operation.
    print("No")

# Call the main solving function.
solve()