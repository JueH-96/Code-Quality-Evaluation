import functools

def solve():
    n = int(input())
    people_data = []
    for i in range(1, n + 1):
        a, b = map(int, input().split())
        people_data.append({'a': a, 'b': b, 'id': i})

    def compare_people(person1, person2):
        a1 = person1['a']
        b1 = person1['b']
        id1 = person1['id']
        a2 = person2['a']
        b2 = person2['b']
        id2 = person2['id']

        v1 = a1 * (a2 + b2)
        v2 = a2 * (a1 + b1)

        if v1 > v2:
            return -1
        elif v1 < v2:
            return 1
        else:
            if id1 < id2:
                return -1
            elif id1 > id2:
                return 1
            else:
                return 0

    sorted_people = sorted(people_data, key=functools.cmp_to_key(compare_people))
    
    output_ids = [str(person['id']) for person in sorted_people]
    print(*(output_ids))

if __name__ == '__main__':
    solve()