import sys

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(N):
        C, P, *S = map(int, sys.stdin.readline().split())
        wheels.append((C, P, S))
    return N, M, wheels

def calculate_expected_value(N, M, wheels):
    # Initialize a dictionary to store the expected values for each state
    expected_values = {0: 0}

    # Iterate over the states in reverse order
    for i in range(M - 1, -1, -1):
        # Initialize the minimum expected value for the current state
        min_expected_value = float('inf')

        # Iterate over the wheels
        for C, P, S in wheels:
            # Calculate the expected value for the current wheel
            expected_value = C
            for s in S:
                next_state = i + s
                if next_state >= M:
                    expected_value += (P - 1) / P * expected_values[i]
                else:
                    expected_value += (1 / P) * (expected_values[next_state] + C)

            # Update the minimum expected value
            min_expected_value = min(min_expected_value, expected_value)

        # Store the minimum expected value for the current state
        expected_values[i] = min_expected_value

    # Return the expected value for the initial state
    return expected_values[0]

def main():
    N, M, wheels = read_input()
    expected_value = calculate_expected_value(N, M, wheels)
    print(expected_value)

if __name__ == "__main__":
    main()