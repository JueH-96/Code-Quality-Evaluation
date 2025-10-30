# YOUR CODE HERE
def insert_numbers(sequence):
    while True:
        for i in range(len(sequence) - 1):
            if abs(sequence[i] - sequence[i+1]) != 1:
                if sequence[i] < sequence[i+1]:
                    new_numbers = list(range(sequence[i] + 1, sequence[i+1]))
                else:
                    new_numbers = list(range(sequence[i] - 1, sequence[i+1], -1))
                sequence[i+1:i+1] = new_numbers
                break
        else:
            return sequence

N = int(input())
A = list(map(int, input().split()))

result = insert_numbers(A)
print(*result)