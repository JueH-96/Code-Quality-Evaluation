import sys

def solve():
    N, M, Q = map(int, input().split())
    people = []
    for _ in range(M):
        S, T = map(int, input().split())
        people.append((S, T))
    
    for _ in range(Q):
        L, R = map(int, input().split())
        
        # Check if it's possible to satisfy the requirements for people L to R
        possible = True
        for i in range(L-1, R):
            S, T = people[i]
            if abs(S - T) <= 1:
                possible = False
                break
            
            # Check if the person can reach T with stamina 0
            stamina = 0
            for j in range(min(S, T)-1, max(S, T)-1):
                stamina += 1 if j+1 < S else -1
            if stamina != 0:
                possible = False
                break
        
        print("Yes" if possible else "No")

if __name__ == "__main__":
    solve()