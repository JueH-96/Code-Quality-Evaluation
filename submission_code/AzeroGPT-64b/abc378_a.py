# standard library packages
# None

# third party packages
# None

# local project packages
# None

def main():
    A = list(map(int, input().split()))
    colors = [0] * 5

    for a in A:
        colors[a] += 1
    
    max_count = max(colors[1:])

    # No pairs if max count is 1 (all balls are of different colors)
    if max_count == 1:
        print(0)
    else:
        # Calculate the maximum number of pairs we can make
        # If max count is 2 or 3, we can make 1 pair
        # If max count is 4, we can make 2 pairs
        pairs = (max_count + 1) // 2
        print(pairs)

# check function to verify the solution with provided data points
def check():
    import io, sys
    # Redirect stdout to capture the output
    capturedOutput = io.StringIO()         
    sys.stdout = capturedOutput

    # Test case inputs
    inputs = ["2 1 2 1", "4 4 4 1", "1 2 3 4"]
    expected_outputs = ["2", "1", "0"]

    # Check each test case
    for i, input_values in enumerate(inputs):
        sys.stdin = io.StringIO(input_values)
        main()
        sys.stdout = sys.__stdout__  # Reset redirect.
        output = capturedOutput.getvalue().strip()
        assert output == expected_outputs[i], f"Error on test case {i}: expected {expected_outputs[i]}, got {output}"
    print("All test cases passed.")

# Uncomment the following line to test the solution
# check()