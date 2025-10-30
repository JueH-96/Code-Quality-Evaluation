import sys

def transform_string(S):
    lowercase_count = sum(1 for c in S if c.islower())
    uppercase_count = sum(1 for c in S if c.isupper())

    if uppercase_count > lowercase_count:
        return S.upper()
    else:
        return S.lower()

if __name__ == "__main__":
    input = sys.stdin.read()
    S = input.strip()
    result = transform_string(S)
    sys.stdout.write(result + '
')