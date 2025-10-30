import sys

def main():
    N, M = map(int, sys.stdin.readline().split())

    # Adjacency matrix: stronger_than[i][j] is True if i is known to be stronger than j.
    # Programmers are 0-indexed internally, from 0 to N-1.
    stronger_than = [[False for _ in range(N)] for _ in range(N)]

    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        # Adjust to 0-indexed
        u, v = A - 1, B - 1
        stronger_than[u][v] = True

    # Floyd-Warshall algorithm for transitive closure.
    # After this, stronger_than[i][j] is true if i is provably stronger than j.
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if stronger_than[i][k] and stronger_than[k][j]:
                    stronger_than[i][j] = True
    
    determined_strongest_person_id = -1

    # Check each person to see if they are provably stronger than all others.
    for p_idx in range(N):  # p_idx is the candidate for the strongest person
        count_weaker_than_p = 0
        for other_idx in range(N):
            if p_idx == other_idx:
                continue
            
            if stronger_than[p_idx][other_idx]:
                count_weaker_than_p += 1
        
        if count_weaker_than_p == N - 1:
            # p_idx is proven to be stronger than all N-1 other people.
            # As proven in thought process, if such a person exists, they are unique.
            determined_strongest_person_id = p_idx + 1 # Convert to 1-indexed for output
            break # Found the unique strongest, no need to check further.
            
    sys.stdout.write(str(determined_strongest_person_id) + "
")

if __name__ == '__main__':
    main()