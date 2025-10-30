import sys

def can_eat_all_dishes(dishes):
    for i in range(len(dishes) - 1):
        if dishes[i] == 'sweet' and dishes[i + 1] == 'sweet':
            return 'No'
    return 'Yes'

if __name__ == "__main__":
    input_lines = sys.stdin.read().splitlines()
    N = int(input_lines[0])
    dishes = input_lines[1:]
    print(can_eat_all_dishes(dishes))