from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    children_dict = defaultdict(list)
    front_person = -1
    for i in range(1, N+1):
        if A[i-1] == -1:
            front_person = i
        else:
            parent = A[i-1]
            children_dict[parent].append(i)
    
    if front_person == -1:
        print("No front person found")
        return
    
    order = []
    queue = deque([front_person])
    while queue:
        person = queue.popleft()
        order.append(person)
        if person in children_dict:
            queue.extend(children_dict[person])
    
    print(' '.join(map(str, order)))

if __name__ == "__main__":
    main()