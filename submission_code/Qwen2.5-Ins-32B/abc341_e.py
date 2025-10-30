import sys

def flip_string(s, L, R):
    for i in range(L, R + 1):
        s[i] = '1' if s[i] == '0' else '0'

def is_good_string(s, L, R):
    for i in range(L, R):
        if s[i] == s[i + 1]:
            return False
    return True

def process_queries(N, Q, S, queries):
    S = list(S)
    for query in queries:
        query_type, L, R = query
        L -= 1  # Adjust for 0-indexing
        if query_type == 1:
            flip_string(S, L, R)
        else:
            if is_good_string(S, L, R):
                print("Yes")
            else:
                print("No")

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N, Q = int(data[index]), int(data[index + 1])
    index += 2
    S = data[index]
    index += 1
    queries = []
    for _ in range(Q):
        query_type, L, R = int(data[index]), int(data[index + 1]), int(data[index + 2])
        queries.append((query_type, L, R))
        index += 3
    
    process_queries(N, Q, S, queries)