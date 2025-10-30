def main():
    import sys
    input_data = sys.stdin.read().strip()
    N = int(input_data)

    # We want a total of 2*N + 1 characters, alternating starting with '1'
    result = []
    for i in range(2 * N + 1):
        if i % 2 == 0:
            result.append('1')
        else:
            result.append('0')
    
    print(''.join(result))

# Do not forget to call main function
main()