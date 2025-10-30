def process_queries(N, Q, S, queries):
    S = list(S)  # Convert string to a list for mutability
    results = []
    
    for query in queries:
        parts = list(map(int, query.split()))
        type_query = parts[0]
        L = parts[1] - 1  # Convert to 0-based index
        R = parts[2] - 1  # Convert to 0-based index
        
        if type_query == 1:
            # Flip characters from L to R
            for i in range(L, R + 1):
                S[i] = '0' if S[i] == '1' else '1'
        elif type_query == 2:
            # Check if the substring S[L:R+1] is a good string
            is_good = True
            for i in range(L, R):
                if S[i] == S[i + 1]:
                    is_good = False
                    break
            results.append("Yes" if is_good else "No")
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, Q = map(int, data[0].split())
    S = data[1]
    queries = data[2:]
    
    results = process_queries(N, Q, S, queries)
    print("
".join(results))

if __name__ == "__main__":
    main()