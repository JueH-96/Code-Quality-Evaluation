def can_shout_takoyaki(a, b, c):
    """
    Determine whether Takahashi can shout his love for takoyaki every day.

    Args:
    a (int): The hour when Takahashi needs to shout his love for takoyaki.
    b (int): The hour when Takahashi goes to bed.
    c (int): The hour when Takahashi wakes up.

    Returns:
    str: "Yes" if Takahashi can shout his love for takoyaki every day, "No" otherwise.
    """
    # Check if Takahashi is awake at the time he needs to shout his love for takoyaki
    if (b <= a <= c) or (b > c and (a >= b or a <= c)):
        return "Yes"
    else:
        return "No"

# Read the input from stdin
a, b, c = map(int, input().split())

# Call the function and print the result
print(can_shout_takoyaki(a, b, c))