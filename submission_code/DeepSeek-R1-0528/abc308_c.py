import sys
from functools import cmp_to_key

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    people = []
    index = 1
    for i in range(1, n + 1):
        a, b = map(int, data[index].split())
        index += 1
        people.append((a, b, i))
    
    def cmp_func(i, j):
        a_i, b_i, idx_i = i
        a_j, b_j, idx_j = j
        product1 = a_i * b_j
        product2 = a_j * b_i
        
        if product1 > product2:
            return -1
        elif product1 < product2:
            return 1
        else:
            if idx_i < idx_j:
                return -1
            elif idx_i > idx_j:
                return 1
            else:
                return 0
                
    sorted_people = sorted(people, key=cmp_to_key(cmp_func))
    result = [str(person[2]) for person in sorted_people]
    print(" ".join(result))

if __name__ == "__main__":
    main()