import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    S = list(data[index])
    index += 1
    
    queries = []
    for _ in range(Q):
        qtype = int(data[index])
        index += 1
        L = int(data[index]) - 1
        index += 1
        R = int(data[index]) - 1
        index += 1
        queries.append((qtype, L, R))
    
    results = []
    
    for qtype, L, R in queries:
        if qtype == 1:
            # Flip the bits from L to R
            for i in range(L, R + 1):
                S[i] = '0' if S[i] == '1' else '1'
        elif qtype == 2:
            # Check if the substring S[L:R+1] is a good string
            good = True
            for i in range(L, R):
                if S[i] == S[i + 1]:
                    good = False
                    break
            results.append("Yes" if good else "No")
    
    # Print all results for type 2 queries
    print("
".join(results))

if __name__ == "__main__":
    main()