import sys

def generate_alternating_string(N):
    result = []
    for i in range(N):
        result.append('1')
        result.append('0')
    result.append('1')
    return ''.join(result)

def main():
    input = sys.stdin.read()
    N = int(input.strip())
    output = generate_alternating_string(N)
    sys.stdout.write(output + '
')

if __name__ == "__main__":
    main()