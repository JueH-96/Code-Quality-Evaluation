def solution(S: list[int]) -> int:
    total = sum(S)
    if len(S) <= 1:
        return total
    local_mins = []
    for i in range(1, len(S)-1):
        if S[i] <= S[i-1] and S[i] <= S[i+1]:
            local_mins.append(S[i])
    if not local_mins:
        return total
    min_local = min(local_mins)
    # Check if the minimal local minimum is the first element
    if local_mins[0] == S[0]:
        # Find the next minimal local minimum
        next_min_local = min([m for m in local_mins if m != S[0]])
        if next_min_local is not None:
            return total - next_min_local
    return total - min_local