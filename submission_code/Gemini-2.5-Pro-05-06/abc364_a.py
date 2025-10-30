import sys

def main():
    N = int(sys.stdin.readline())
    S = []
    for _ in range(N):
        S.append(sys.stdin.readline().strip())

    can_eat_all = True
    
    # We check for pairs (S[i], S[i+1]) that are both "sweet".
    # If such a pair exists, Takahashi eats S[i+1] and then feels sick.
    # He cannot eat S[i+2].
    # If S[i+2] was one of the N dishes, he fails to eat all dishes.
    # S[i+2] exists if i+2 < N, which means i < N-2.
    # So, we loop i from 0 to N-3 (inclusive).
    # In Python, this loop is range(N-2).
    # If N <= 2, range(N-2) is empty, so the loop doesn't run.
    # This correctly implies "Yes" for N=1 or N=2, as he either doesn't
    # encounter two consecutive sweet dishes, or feels sick after the last one.
    for i in range(N - 2):  # i iterates from 0 to N-3
        if S[i] == "sweet" and S[i+1] == "sweet":
            can_eat_all = False
            break
            
    if can_eat_all:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()