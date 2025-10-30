import sys

def can_divide_into_equal_sum_groups(A, B, C):
    total_sum = A + B + C

    # Check if the total sum can be divided into equal parts
    if total_sum % 2 == 0:
        target_sum = total_sum // 2
        # Check if we can form two groups with equal sum
        if (A == target_sum or B == target_sum or C == target_sum or
            A + B == target_sum or A + C == target_sum or B + C == target_sum):
            return "Yes"

    # Check if the total sum can be divided into three equal parts
    if total_sum % 3 == 0:
        target_sum = total_sum // 3
        # Check if we can form three groups with equal sum
        if A == target_sum and B == target_sum and C == target_sum:
            return "Yes"

    return "No"

def main():
    input = sys.stdin.read()
    A, B, C = map(int, input.strip().split())
    result = can_divide_into_equal_sum_groups(A, B, C)
    sys.stdout.write(result + "
")

if __name__ == "__main__":
    main()