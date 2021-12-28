####################################################################################################
#########################  EXACT DIAGONALIZATION - HUBBARD MODEL  ##################################
####################################################################################################


import numpy as np
import itertools


####################################################################################################
########################################  Helper Functions  ########################################
####################################################################################################

''' turn integer to binary represented as a string '''
def binary(num):
    return bin(num).replace("0b", "")


''' pad bits with zeros to left to adjust the size to number of sites '''
def push_zero(str, nsites):
    # add zero until desired size is obtained
    while abs(nsites - len(str)) > 0:
        str = '0' + str

    return str


''' find all permutations for given number of sites and electrons '''
def all_permutations(nsites, spin):
    return list(set(itertools.permutations([1]*spin + [0]*(nsites-spin))))


''' count annihilation/destruction operators to get correct fermionic sign '''
def fermionic_sign(left, right, nsites):
    # XOR
    new1 = left[0] ^ right[0]
    new2 = left[1] ^ right[1]

    # if spin up part is zero
    if new1 == 0:
        sign1 = push_zero(binary(new2 & left[1]), nsites)
        sign2 = push_zero(binary(new2 & right[1]), nsites)
        index1 = sign1.index('1')
        index2 = sign2.index('1')

        cnt1 = binary(left[0]).count('1')
        cnt1 += push_zero(binary(left[1]), nsites).count('1', 0, index1)
        cnt2 = binary(right[0]).count('1')
        cnt2 += push_zero(binary(right[1]), nsites).count('1', 0, index2)

        return cnt1 + cnt2

    # if spin down part is zero
    elif new2 == 0:
        sign1 = push_zero(binary(new1 & left[0]), nsites)
        sign2 = push_zero(binary(new1 & right[0]), nsites)
        index1 = sign1.index('1')
        index2 = sign2.index('1')

        cnt1 = push_zero(binary(left[0]), nsites).count('1', 0, index1)
        cnt2 = push_zero(binary(right[0]), nsites).count('1', 0, index2)

        return cnt1 + cnt2



####################################################################################################
#####################################  Get Hamiltonian  ############################################
####################################################################################################

''' get interaction part of hamiltonian per block: <left|H_int|right> '''
def interaction(left, right, u):
    # only diagonal terms contribute
    if left == right:
        overlap = binary(left[0] & left[1])
        cnt = overlap.count('1')
        return cnt * u
    # off-diagonals are zero
    else:
        return 0.


''' get hopping part of hamiltonian per block: <left|H_hop|right> '''
def hopping(left, right, t, nsites):
    # only off-diagonal terms contribute
    if left != right:
        # XOR
        overlap1 = left[0] ^ right[0]
        overlap2 = left[1] ^ right[1]

        # if spin up part is 0
        if overlap1 == 0:
            # check if hopping to nn possible
            overlap_bits = binary(overlap2)
            if overlap_bits.count('1') == 2:
                # periodic boundaries
                if overlap_bits[0] == '1' and overlap_bits[-1] == '1':
                    cnt = fermionic_sign(left, right, nsites)
                    return -t * (-1.)**cnt

                # neighbouring 1s means hopping is possible
                for i in range(len(overlap_bits)-1):
                    if overlap_bits[i] == '1' and overlap_bits[i+1] == '1':
                        cnt = fermionic_sign(left, right, nsites)
                        return -t * (-1.)**cnt
                    else:
                        return 0.

            else:
                return 0.

        # if spin down part is 0
        elif overlap2 == 0:
            # check if hopping to nn possible
            overlap_bits = binary(overlap1)
            if overlap_bits.count('1') == 2:
                # periodic boundaries
                if overlap_bits[0] == '1' and overlap_bits[-1] == '1':
                    cnt = fermionic_sign(left, right, nsites)
                    return -t * (-1.)**cnt

                # neighbouring 1s means hopping is possible
                for i in range(len(overlap_bits)-1):
                    if overlap_bits[i] == '1' and overlap_bits[i+1] == '1':
                        cnt = fermionic_sign(left, right, nsites)
                        return -t * (-1.)**cnt
                    else:
                        return 0.

            else:
                return 0.

        else:
            return 0.

    # return 0 if neither spin up nor down are 0 after XOR
    else:
        return 0.


''' set basis vectors |spin up>|spin down> '''
def basis(block, nsites):
    # basis to store vectors as a tuple (spin up, spin down)
    basis = []

    # construct spin up vectors
    left_vecs = []
    left = all_permutations(nsites, block[0])
    for i in range(len(left)):
        binary = ""
        for j in range(nsites):
            binary += str(left[i][j])
        left_vecs.append(int(binary, 2))

    # construct spin down vectors
    right_vecs = []
    right = all_permutations(nsites, block[1])
    for i in range(len(right)):
        binary = ""
        for j in range(nsites):
            binary += str(right[i][j])
        right_vecs.append(int(binary, 2))

    # get all combinations of right and left vecs
    for i in range(len(left)):
        for j in range(len(right)):
            basis.append((left_vecs[i], right_vecs[j]))

    return basis


''' get the full hamiltonian per block '''
def get_hamiltonian(block, nsites, u, t):
    # basis vectors
    vecs = basis(block, nsites)
    n = len(vecs)

    # empty hamiltonian to fill
    H = np.zeros([n,n], dtype = float)

    # fill hamiltonian
    for i in range(n):
        for j in range(n):
            H[i,j] += hopping(vecs[i], vecs[j], t, nsites)
            H[i,j] += interaction(vecs[i], vecs[j], u)

    return H



####################################################################################################
########################################  Driver Program  ##########################################
####################################################################################################

def main():
    u = 3.
    t = 1.
    nsites = 6
    block = (nsites//2, nsites//2)

    # create hamiltonian and print eigenvalues
    H = get_hamiltonian(block, nsites, u, t)
    print(np.linalg.eigvalsh(H))


if __name__ == "__main__":
    main()
