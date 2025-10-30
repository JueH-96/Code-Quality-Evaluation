# YOUR CODE HERE
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.value_to_node = {}

    def append(self, value):
        new_node = Node(value)
        self.value_to_node[value] = new_node
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def insert_after(self, x, y):
        node_x = self.value_to_node[x]
        new_node = Node(y)
        self.value_to_node[y] = new_node
        new_node.next = node_x.next
        new_node.prev = node_x
        if node_x.next:
            node_x.next.prev = new_node
        node_x.next = new_node
        if node_x == self.tail:
            self.tail = new_node

    def remove(self, x):
        node = self.value_to_node[x]
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        del self.value_to_node[x]

    def to_list(self):
        result = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next
        return result

N = int(input())
A = list(map(int, input().split()))
Q = int(input())

linked_list = LinkedList()
for value in A:
    linked_list.append(value)

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        linked_list.insert_after(query[1], query[2])
    else:
        linked_list.remove(query[1])

print(*linked_list.to_list())