import sys

def main():
    data = sys.stdin.read().split()
    nums = list(map(int, data))
    N = nums[0]
    index = 1
    people = []
    for i in range(1, N + 1):
        A = nums[index]
        B = nums[index + 1]
        index += 2
        rate = A / (A + B)
        people.append((-rate, i))
    people.sort()
    result = [str(person[1]) for person in people]
    print(' '.join(result))

if __name__ == '__main__':
    main()