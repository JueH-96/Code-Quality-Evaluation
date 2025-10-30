# YOUR CODE HERE
def convert_case(s):
    upper_count = sum(1 for char in s if char.isupper())
    lower_count = sum(1 for char in s if char.islower())
    
    if upper_count > lower_count:
        return s.upper()
    else:
        return s.lower()

if __name__ == "__main__":
    import sys
    input_string = sys.stdin.read().strip()
    result = convert_case(input_string)
    print(result)