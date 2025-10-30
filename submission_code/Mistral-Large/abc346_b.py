import sys

def can_form_substring(W, B):
    # The pattern "wbwbwwbwbwbw" repeats every 12 characters.
    # We need to check if we can form a substring with W 'w's and B 'b's within this repeating pattern.

    # The repeating pattern has 7 'w's and 5 'b's.
    pattern = "wbwbwwbwbwbw"
    count_w = pattern.count('w')
    count_b = pattern.count('b')

    # Calculate the ratio of 'w's to 'b's in the pattern.
    ratio_w = W / (W + B)
    ratio_b = B / (W + B)

    # Calculate the ratio of 'w's to 'b's in the repeating pattern.
    pattern_ratio_w = count_w / (count_w + count_b)
    pattern_ratio_b = count_b / (count_w + count_b)

    # Check if the ratios match.
    if ratio_w == pattern_ratio_w and ratio_b == pattern_ratio_b:
        return "Yes"
    else:
        return "No"

def main():
    input = sys.stdin.read()
    W, B = map(int, input.strip().split())
    result = can_form_substring(W, B)
    sys.stdout.write(result + '
')

if __name__ == "__main__":
    main()