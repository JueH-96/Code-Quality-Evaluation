# YOUR CODE HERE
import sys
input = sys.stdin.read

def process_queries(N, Q, S, queries):
    # Convert S to a list for easier manipulation
    S = list(S)
    results = []
    
    for query in queries:
        parts = query.split()
        query_type = int(parts[0])
        L = int(parts[1]) - 1  # Convert to 0-based index
        R = int(parts[2]) - 1  # Convert to 0-based index
        
        if query_type == 1:
            # Flip each of the L-th to R-th characters of S
            for i in range(L, R + 1):
                S[i] = '0' if S[i] == '1' else '1'
        
        elif query_type == 2:
            # Check if the substring S[L:R+1] is a good string
            is_good = True
            for i in range(L, R):
                if S[i] == S[i + 1]:
                    is_good = False
                    break
            results.append("Yes" if is_good else "No")
    
    return results

def main():
    data = input().strip().split('
')
    N, Q = map(int, data[0].split())
    S = data[1].strip()
    queries = data[2:]
    
    results = process_queries(N, Q, S, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()