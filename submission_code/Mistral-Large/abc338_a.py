import sys

def check_string_format(S):
    if S[0].isupper() and S[1:].islower():
        return "Yes"
    else:
        return "No"

if __name__ == "__main__":
    input = sys.stdin.read()
    S = input.strip()
    result = check_string_format(S)
    sys.stdout.write(result + "
")