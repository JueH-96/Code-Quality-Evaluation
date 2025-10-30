def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    count = 0  # to count current consecutive healthy teeth
    answer = 0

    for tooth in S:
        if tooth == 'O':
            count += 1
        else:
            count = 0  # reset the count if the tooth is unhealthy

        if count == K:
            answer += 1
            count = 0  # after eating a strawberry, mark these K teeth as unhealthy (reset count)

    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()