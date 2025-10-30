def main():
    N = int(input())
    result = []
    for i in range(1, N + 1):
        # If i is a multiple of 3, it's a failure ('x'), otherwise a success ('o')
        result.append('x' if i % 3 == 0 else 'o')
    print(''.join(result))

main()