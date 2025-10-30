def find_staring_bibs(N, P, Q):
    # Create a mapping from person index to bib number
    bib_map = {}
    for i in range(N):
        bib_map[P[i]] = Q[i]
    
    # Prepare the result based on the staring person
    result = []
    for i in range(1, N + 1):
        staring_person = P[i - 1]  # P_i is 1-indexed, so we use i-1
        result.append(bib_map[staring_person])
    
    return result

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    P = list(map(int, data[1].split()))
    Q = list(map(int, data[2].split()))
    
    result = find_staring_bibs(N, P, Q)
    print(" ".join(map(str, result)))