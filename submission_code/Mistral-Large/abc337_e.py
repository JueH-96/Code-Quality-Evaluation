import sys
import math

def find_spoiled_bottle():
    # Read the number of bottles
    N = int(sys.stdin.readline().strip())

    # Calculate the minimum number of friends needed
    M = math.ceil(math.log2(N))
    print(M)

    # Generate the distribution of bottles to friends
    for i in range(M):
        # Each friend will test a subset of bottles
        K_i = 1 << i  # 2^i
        A_i = [j for j in range(1, N + 1) if (j & (1 << i)) != 0]
        print(len(A_i), *A_i)

    sys.stdout.flush()

    # Read the result string from the judge
    S = sys.stdin.readline().strip()

    # Determine the spoiled bottle number
    spoiled_bottle = 0
    for i in range(M):
        if S[i] == '1':
            spoiled_bottle |= (1 << i)

    print(spoiled_bottle)
    sys.stdout.flush()

if __name__ == "__main__":
    find_spoiled_bottle()