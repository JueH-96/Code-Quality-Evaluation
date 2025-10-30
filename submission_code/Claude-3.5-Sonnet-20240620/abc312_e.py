# YOUR CODE HERE
def intersect_1d(a1, a2, b1, b2):
    return max(a1, b1) <= min(a2, b2)

def share_face(c1, c2):
    dx = intersect_1d(c1[0], c1[3], c2[0], c2[3])
    dy = intersect_1d(c1[1], c1[4], c2[1], c2[4])
    dz = intersect_1d(c1[2], c1[5], c2[2], c2[5])
    
    face_area = 0
    if dx and dy and c1[2] == c2[5] or c1[5] == c2[2]:
        face_area = (min(c1[3], c2[3]) - max(c1[0], c2[0])) * (min(c1[4], c2[4]) - max(c1[1], c2[1]))
    elif dx and dz and c1[1] == c2[4] or c1[4] == c2[1]:
        face_area = (min(c1[3], c2[3]) - max(c1[0], c2[0])) * (min(c1[5], c2[5]) - max(c1[2], c2[2]))
    elif dy and dz and c1[0] == c2[3] or c1[3] == c2[0]:
        face_area = (min(c1[4], c2[4]) - max(c1[1], c2[1])) * (min(c1[5], c2[5]) - max(c1[2], c2[2]))
    
    return face_area > 0

N = int(input())
cuboids = [list(map(int, input().split())) for _ in range(N)]

shared_faces = [0] * N

for i in range(N):
    for j in range(N):
        if i != j and share_face(cuboids[i], cuboids[j]):
            shared_faces[i] += 1

for count in shared_faces:
    print(count)