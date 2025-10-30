def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    A = int(input_data[0])
    B = int(input_data[1])
    
    suspects = {1, 2, 3}
    possible_culprits = suspects - {A, B}
    
    if len(possible_culprits) == 1:
        print(possible_culprits.pop())
    else:
        print(-1)

if __name__ == '__main__':
    main()