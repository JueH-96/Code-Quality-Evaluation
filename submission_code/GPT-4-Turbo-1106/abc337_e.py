import math

def main():
    # Read the number of bottles from stdin
    N = int(input().strip())

    # Calculate the number of friends needed using binary representation
    M = math.ceil(math.log2(N))

    # Print the number of friends
    print(M)

    # Distribute the bottles to the friends using binary representation
    for i in range(M):
        # Determine which bottles to give to the i-th friend
        bottles = [str(j) for j in range(1, N + 1) if (j >> i) & 1]
        # Print the number of bottles and their numbers
        print(len(bottles), ' '.join(bottles))

    # Flush to ensure the output is sent to the judge
    import sys
    sys.stdout.flush()

    # Read the judge's response
    S = input().strip()

    # Determine the spoiled bottle's number using the responses
    spoiled_bottle = 0
    for i in range(M):
        if S[i] == '1':
            spoiled_bottle |= 1 << i

    # Print the spoiled bottle's number
    print(spoiled_bottle)

if __name__ == "__main__":
    main()