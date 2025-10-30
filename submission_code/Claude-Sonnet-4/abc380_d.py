def solve():
    S = input().strip()
    Q = int(input())
    queries = list(map(int, input().split()))
    
    def get_char(pos, length, flip_count):
        # pos is 1-indexed
        if length == len(S):
            # Base case: we're at the original string
            char = S[pos - 1]
            if flip_count % 2 == 1:
                if char.islower():
                    return char.upper()
                else:
                    return char.lower()
            return char
        
        half_length = length // 2
        if pos <= half_length:
            # In the first half (original part)
            return get_char(pos, half_length, flip_count)
        else:
            # In the second half (flipped part)
            return get_char(pos - half_length, half_length, flip_count + 1)
    
    results = []
    for k in queries:
        # Find the minimum length that can contain position k
        length = len(S)
        while length < k:
            length *= 2
        
        result = get_char(k, length, 0)
        results.append(result)
    
    print(' '.join(results))

solve()