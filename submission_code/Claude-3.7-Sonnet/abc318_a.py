# YOUR CODE HERE
N, M, P = map(int, input().split())

if M > N:
    # If the first full moon is after day N, there are no full moons
    print(0)
else:
    # Calculate the number of full moons from day M to day N
    # This is the number of terms in the arithmetic sequence M, M+P, M+2P, ..., M+kP where M+kP <= N
    full_moon_count = (N - M) // P + 1
    print(full_moon_count)