def solve():
    N, M, Q = map(int, input().strip().split())
    
    # Read the information about each person
    people = []
    for _ in range(M):
        S, T = map(int, input().strip().split())
        people.append((S, T))
    
    # Read the queries
    queries = []
    for _ in range(Q):
        L, R = map(int, input().strip().split())
        queries.append((L, R))
    
    for L, R in queries:
        # Check if the constraints of people L to R can be satisfied
        subset_people = people[L-1:R]
        if is_viable(subset_people):
            print("Yes")
        else:
            print("No")

def is_viable(subset_people):
    """
    Check if there's a way to set road strengths to satisfy all people's constraints.
    
    Two people have conflicting constraints if they travel along the same road segment
    but in opposite directions.
    """
    for i in range(len(subset_people)):
        for j in range(i + 1, len(subset_people)):
            S1, T1 = subset_people[i]
            S2, T2 = subset_people[j]
            
            # Check if the two people travel in opposite directions
            direction1 = 1 if S1 < T1 else -1
            direction2 = 1 if S2 < T2 else -1
            
            if direction1 * direction2 < 0:  # Opposite directions
                # Check if their paths overlap
                path1 = set(range(min(S1, T1), max(S1, T1)))
                path2 = set(range(min(S2, T2), max(S2, T2)))
                
                if path1.intersection(path2):
                    return False
    
    return True

if __name__ == "__main__":
    solve()