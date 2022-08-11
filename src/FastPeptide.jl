module FastPeptide
using Random

export calc_binding_strengths, stringhamming, fasthamming, str2matr

function str2peptide(str::String, len::Int)
    pep = 0
    i = len
    @inbounds while i >= 1
        pep = (pep << AABITS) | aa2code[str[i] - 'A' + 1]
        i -= 1
    end
    return pep
end

function read_peptides(peptstrings::Array{String})
    peptides = Array{Int,1}(undef, 0)
    @inbounds for str in peptstrings
        push!(peptides, str2peptide(str, length(str)))
    end
    return peptides
end

function whamming(p1::Int, p2::Int, peplen::Int)
    dist = 1.0
    @inbounds for i in 1:peplen
        dist *= binding_matrices[i][(p1 & AAMASK) + 1][(p2 & AAMASK) + 1]
        p1 >>= AABITS
        p2 >>= AABITS
    end
    dist = dist^(1.0/peplen)
    return Int16(trunc(dist))
end

function calc_binding_strengths(peps::Array{String}, tcrs::Array{String})
    #bindingarray = Dict{String, Int16}()
    d = 0
    @inbounds for i in 1:length(peps)
        p1 = peps[i]
        peplen = length(p1)
        @inbounds for j in 1:length(tcrs)
            t1 = tcrs[j]
            d = whamming(str2peptide(t1, length(t1)), str2peptide(p1, length(p1)), peplen)
            #bindingarray[i] = d
            #if d >= 100
        end
    end
    return d
    #return bindingarray
end

function calc_binding_strengths(peps::String, tcrs::String)
    #bindingarray = Dict{String, Int16}()
    d = 0
    peplen = length(peps)
    tcrlen = length(tcrs)
    d = whamming(str2peptide(tcrs, tcrlen), str2peptide(peps, peplen), peplen)
    return d
    #return bindingarray
end

function str2matr(st)
    s = split(st,"")
    s = Array{String}(s)
    s = reshape(s, 1, length(s))
end

function hamming(A::BitArray, B::BitArray)
    #size(A) == size(B) || throw(DimensionMismatch("sizes of A and B must match"))
    Ac,Bc = A.chunks, B.chunks
    W = 0
    for i = 1:(length(Ac)-1)
        W += count_ones(Ac[i] ⊻ Bc[i])
    end
    W += count_ones(Ac[end] ⊻ Bc[end] & Base._msk_end(A))
    return W
end

function stringhamming(A::String, B::String)
    counts = 0
    for i in eachindex(A)
        if A[i] == B[i]
        counts += 1
        end
    end
    return counts
end

function fasthamming(peps::Matrix{Int}, tcr, threshold::Int, reaction_dict::Dict{Array{Int}, Int}, min_strength::Int, finalcheck::Bool)
    res = peps .- tcr
    if size(res)[1] > 1
        matches = sum(x -> x == 0, res, dims=2)
        # Add all of matches for each peptide to thymocyte memory dict
        # reaction_dict[peps[i]] = matches[i]]
        if finalcheck == false
            death_label = false
            for i in 1:size(peps)[1]
                if get(reaction_dict, Matrix(peps[i,:]'), 0) != 0 && matches[i] >= min_strength # if thymocyte has seen antigen before, add to its current reaction level
                    reaction_dict[Matrix(peps[i,:]')] += matches[i]
                elseif matches[i] >= min_strength # otherwise, add antigen as a new entry to the reaction_dict dict
                    reaction_dict[Matrix(peps[i,:]')] = matches[i]
                else
                    continue
                end
                if reaction_dict[Matrix(peps[i,:]')] >= threshold
                    death_label = true
                end
            end
            return death_label
        else
            if any(match >= threshold for match in matches) # can count successful/unsuccessful interactions here maybe
                return true
            else
                return false
            end
        end

    else
        match = sum(x -> x == 0, res)
        if finalcheck == false
            death_label = false
            if get(reaction_dict, peps, 0) != 0 && match >= min_strength # if thymocyte has seen antigen before, add to its current reaction level
                reaction_dict[peps] += match
            elseif match >= min_strength # otherwise, add antigen as a new entry to the reaction_dict dict
                reaction_dict[peps] = match
            else
                return false
            end

            if reaction_dict[peps] >= threshold
                death_label = true
            end

            return death_label
        else
            if match >= threshold
                return true
            else
                return false
            end
        end
    end
end



global const AABITS = 5
global const AAMASK = 31
global const aa2code = [
    0,    21,   1,    2,
    3,    4,    5,    6,
    7,    21,   8,    9,
    10,   11,   21,   12,
    13,   14,   15,   16,
    21,   17,   18,   21,
    19,   21
]

global const binding_matrices = [
[[100, 79, 363, 311, 33, 142, 83, 83, 124, 67, 51, 69, 369, 141, 97, 92, 106, 103, 55, 35, ],
[126, 100, 458, 392, 42, 179, 104, 104, 156, 85, 65, 87, 466, 179, 123, 116, 134, 130, 69, 44, ],
[28, 22, 100, 86, 9, 39, 23, 23, 34, 18, 14, 19, 102, 39, 27, 25, 29, 28, 15, 10, ],
[32, 26, 117, 100, 11, 46, 27, 27, 40, 22, 16, 22, 119, 46, 31, 30, 34, 33, 18, 11, ],
[301, 238, 1093, 935, 100, 426, 248, 249, 372, 201, 154, 208, 1112, 426, 293, 276, 319, 309, 165, 106, ],
[71, 56, 256, 219, 23, 100, 58, 58, 87, 47, 36, 49, 261, 100, 69, 65, 75, 72, 39, 25, ],
[121, 96, 440, 376, 40, 172, 100, 100, 150, 81, 62, 84, 448, 171, 118, 111, 129, 124, 67, 43, ],
[121, 96, 440, 376, 40, 171, 100, 100, 150, 81, 62, 84, 447, 171, 118, 111, 128, 124, 67, 43, ],
[81, 64, 294, 251, 27, 114, 67, 67, 100, 54, 41, 56, 299, 114, 79, 74, 86, 83, 44, 28, ],
[149, 118, 542, 464, 50, 212, 123, 123, 185, 100, 76, 103, 552, 211, 145, 137, 159, 153, 82, 53, ],
[195, 155, 710, 607, 65, 277, 161, 161, 242, 131, 100, 135, 722, 276, 190, 179, 207, 201, 107, 69, ],
[144, 114, 525, 449, 48, 205, 119, 119, 179, 97, 74, 100, 534, 204, 141, 133, 153, 148, 79, 51, ],
[27, 21, 98, 84, 9, 38, 22, 22, 33, 18, 14, 19, 100, 38, 26, 25, 29, 28, 15, 10, ],
[71, 56, 257, 220, 23, 100, 58, 58, 87, 47, 36, 49, 261, 100, 69, 65, 75, 73, 39, 25, ],
[103, 81, 373, 319, 34, 145, 85, 85, 127, 69, 53, 71, 380, 145, 100, 94, 109, 105, 56, 36, ],
[109, 86, 395, 338, 36, 154, 90, 90, 135, 73, 56, 75, 402, 154, 106, 100, 116, 112, 60, 38, ],
[94, 75, 342, 293, 31, 133, 78, 78, 117, 63, 48, 65, 348, 133, 92, 87, 100, 97, 52, 33, ],
[97, 77, 354, 303, 32, 138, 80, 80, 121, 65, 50, 67, 360, 138, 95, 89, 103, 100, 54, 34, ],
[182, 144, 661, 565, 61, 258, 150, 150, 225, 122, 93, 126, 672, 258, 177, 167, 193, 187, 100, 64, ],
[284, 225, 1030, 881, 94, 402, 234, 234, 351, 190, 145, 196, 1048, 401, 276, 261, 301, 291, 156, 100, ]],
[[100, 103, 191, 184, 42, 61, 87, 115, 139, 44, 63, 144, 144, 121, 148, 97, 123, 115, 52, 103, ],
[97, 100, 185, 178, 41, 59, 84, 111, 135, 43, 61, 139, 140, 118, 144, 94, 120, 111, 50, 100, ],
[52, 54, 100, 96, 22, 32, 45, 60, 73, 23, 33, 75, 76, 64, 78, 51, 65, 60, 27, 54, ],
[54, 56, 104, 100, 23, 33, 47, 62, 76, 24, 34, 78, 79, 66, 81, 53, 67, 62, 28, 56, ],
[235, 243, 449, 432, 100, 143, 204, 270, 328, 104, 147, 338, 340, 286, 349, 228, 291, 270, 122, 242, ],
[164, 170, 314, 302, 70, 100, 142, 188, 229, 73, 103, 236, 237, 199, 244, 159, 203, 189, 85, 169, ],
[116, 119, 221, 212, 49, 70, 100, 132, 161, 51, 72, 166, 167, 140, 171, 112, 143, 133, 60, 119, ],
[87, 90, 167, 160, 37, 53, 76, 100, 122, 39, 55, 125, 126, 106, 129, 84, 108, 100, 45, 90, ],
[72, 74, 137, 132, 30, 44, 62, 82, 100, 32, 45, 103, 103, 87, 106, 69, 89, 82, 37, 74, ],
[226, 233, 432, 415, 96, 138, 196, 259, 315, 100, 142, 325, 326, 274, 335, 219, 279, 259, 117, 233, ],
[160, 165, 305, 293, 68, 97, 138, 183, 223, 71, 100, 229, 230, 194, 236, 155, 197, 183, 83, 164, ],
[70, 72, 133, 128, 30, 42, 60, 80, 97, 31, 44, 100, 100, 84, 103, 67, 86, 80, 36, 72, ],
[69, 71, 132, 127, 29, 42, 60, 79, 97, 31, 43, 100, 100, 84, 103, 67, 86, 80, 36, 71, ],
[82, 85, 157, 151, 35, 50, 71, 95, 115, 36, 52, 118, 119, 100, 122, 80, 102, 95, 43, 85, ],
[68, 70, 129, 124, 29, 41, 58, 77, 94, 30, 42, 97, 97, 82, 100, 65, 83, 77, 35, 69, ],
[103, 107, 197, 190, 44, 63, 89, 118, 144, 46, 65, 148, 149, 125, 153, 100, 128, 119, 53, 106, ],
[81, 84, 155, 149, 34, 49, 70, 93, 113, 36, 51, 116, 117, 98, 120, 78, 100, 93, 42, 83, ],
[87, 90, 166, 160, 37, 53, 75, 100, 121, 39, 55, 125, 126, 106, 129, 84, 108, 100, 45, 90, ],
[193, 200, 370, 355, 82, 118, 168, 222, 270, 86, 121, 278, 279, 235, 287, 187, 239, 222, 100, 199, ],
[97, 100, 186, 179, 41, 59, 84, 112, 136, 43, 61, 140, 140, 118, 144, 94, 120, 112, 50, 100, ]],
[[100, 40, 43, 52, 20, 29, 36, 26, 52, 23, 12, 35, 11, 15, 60, 24, 33, 14, 36, 35, ],
[247, 100, 107, 130, 50, 72, 88, 63, 130, 57, 29, 87, 27, 38, 148, 58, 81, 36, 89, 87, ],
[232, 94, 100, 122, 47, 68, 83, 59, 122, 53, 27, 81, 25, 35, 139, 55, 76, 33, 83, 82, ],
[191, 77, 82, 100, 38, 56, 68, 49, 100, 44, 22, 67, 21, 29, 114, 45, 62, 28, 68, 67, ],
[498, 201, 215, 261, 100, 146, 178, 127, 261, 114, 58, 175, 54, 76, 298, 118, 162, 72, 179, 175, ],
[342, 138, 147, 180, 69, 100, 122, 87, 179, 78, 40, 120, 37, 52, 205, 81, 112, 49, 123, 120, ],
[280, 113, 121, 147, 56, 82, 100, 72, 147, 64, 33, 98, 30, 43, 168, 66, 91, 40, 101, 99, ],
[391, 158, 169, 205, 79, 114, 140, 100, 205, 89, 46, 137, 42, 60, 234, 92, 128, 56, 141, 138, ],
[191, 77, 82, 100, 38, 56, 68, 49, 100, 44, 22, 67, 21, 29, 114, 45, 62, 28, 68, 67, ],
[438, 177, 189, 230, 88, 128, 156, 112, 229, 100, 51, 153, 48, 67, 262, 103, 143, 63, 157, 154, ],
[854, 345, 368, 448, 171, 250, 304, 218, 448, 195, 100, 299, 93, 130, 511, 201, 278, 123, 307, 300, ],
[285, 115, 123, 150, 57, 83, 102, 73, 150, 65, 33, 100, 31, 43, 171, 67, 93, 41, 103, 100, ],
[921, 372, 397, 483, 185, 269, 328, 235, 483, 211, 108, 323, 100, 140, 551, 217, 300, 133, 331, 324, ],
[656, 265, 283, 344, 132, 192, 234, 168, 344, 150, 77, 230, 71, 100, 393, 155, 214, 95, 236, 231, ],
[167, 68, 72, 88, 34, 49, 60, 43, 88, 38, 20, 59, 18, 25, 100, 39, 54, 24, 60, 59, ],
[424, 171, 183, 222, 85, 124, 151, 108, 222, 97, 50, 149, 46, 65, 254, 100, 138, 61, 152, 149, ],
[307, 124, 132, 161, 62, 90, 109, 78, 161, 70, 36, 107, 33, 47, 184, 72, 100, 44, 110, 108, ],
[693, 280, 299, 364, 139, 203, 247, 177, 363, 158, 81, 243, 75, 106, 415, 163, 226, 100, 249, 244, ],
[278, 113, 120, 146, 56, 81, 99, 71, 146, 64, 33, 98, 30, 42, 167, 66, 91, 40, 100, 98, ],
[284, 115, 123, 149, 57, 83, 101, 73, 149, 65, 33, 100, 31, 43, 170, 67, 93, 41, 102, 100, ]],
[[100, 95, 526, 170, 126, 520, 186, 81, 155, 90, 113, 254, 374, 111, 105, 98, 115, 101, 131, 96, ],
[106, 100, 556, 180, 133, 550, 197, 86, 164, 96, 119, 269, 396, 117, 111, 103, 122, 107, 138, 102, ],
[19, 18, 100, 32, 24, 99, 35, 15, 30, 17, 21, 48, 71, 21, 20, 19, 22, 19, 25, 18, ],
[59, 56, 309, 100, 74, 306, 109, 48, 91, 53, 66, 149, 220, 65, 62, 57, 68, 60, 77, 57, ],
[79, 75, 417, 135, 100, 413, 148, 65, 123, 72, 89, 201, 297, 88, 84, 77, 91, 80, 104, 76, ],
[19, 18, 101, 33, 24, 100, 36, 16, 30, 17, 22, 49, 72, 21, 20, 19, 22, 19, 25, 19, ],
[54, 51, 283, 91, 68, 279, 100, 44, 83, 49, 61, 136, 201, 59, 57, 52, 62, 54, 70, 52, ],
[123, 116, 647, 209, 155, 640, 229, 100, 191, 111, 139, 312, 461, 136, 130, 120, 142, 125, 161, 118, ],
[64, 61, 339, 110, 81, 335, 120, 52, 100, 58, 73, 164, 241, 71, 68, 63, 74, 65, 84, 62, ],
[111, 104, 581, 188, 139, 574, 206, 90, 172, 100, 125, 281, 414, 122, 116, 108, 127, 112, 144, 106, ],
[89, 84, 466, 151, 112, 461, 165, 72, 138, 80, 100, 225, 332, 98, 93, 87, 102, 90, 116, 85, ],
[39, 37, 207, 67, 50, 205, 73, 32, 61, 36, 44, 100, 148, 44, 41, 38, 45, 40, 51, 38, ],
[27, 25, 140, 45, 34, 139, 50, 22, 41, 24, 30, 68, 100, 30, 28, 26, 31, 27, 35, 26, ],
[90, 85, 475, 154, 114, 470, 168, 73, 140, 82, 102, 229, 339, 100, 95, 88, 104, 92, 118, 87, ],
[95, 90, 499, 162, 120, 494, 177, 77, 147, 86, 107, 241, 356, 105, 100, 93, 110, 96, 124, 91, ],
[103, 97, 539, 174, 129, 533, 191, 83, 159, 93, 116, 260, 384, 113, 108, 100, 118, 104, 134, 99, ],
[87, 82, 456, 148, 109, 451, 161, 71, 135, 78, 98, 220, 325, 96, 91, 85, 100, 88, 113, 84, ],
[99, 93, 519, 168, 124, 513, 184, 80, 153, 89, 111, 251, 370, 109, 104, 96, 114, 100, 129, 95, ],
[77, 72, 402, 130, 96, 398, 142, 62, 119, 69, 86, 194, 286, 85, 81, 75, 88, 78, 100, 74, ],
[104, 98, 546, 177, 131, 540, 193, 84, 161, 94, 117, 264, 389, 115, 109, 101, 120, 105, 136, 100, ]],
[[100, 208, 217, 302, 129, 121, 102, 210, 190, 66, 78, 47, 378, 207, 224, 79, 239, 198, 342, 347, ],
[48, 100, 105, 145, 62, 58, 49, 101, 91, 32, 38, 23, 182, 100, 108, 38, 115, 95, 165, 167, ],
[46, 96, 100, 139, 59, 56, 47, 97, 87, 30, 36, 22, 174, 95, 103, 36, 110, 91, 158, 160, ],
[33, 69, 72, 100, 43, 40, 34, 70, 63, 22, 26, 16, 125, 69, 74, 26, 79, 65, 113, 115, ],
[78, 161, 168, 234, 100, 94, 79, 163, 147, 51, 61, 37, 293, 161, 174, 61, 185, 153, 265, 269, ],
[83, 172, 180, 250, 107, 100, 84, 174, 157, 54, 65, 39, 312, 171, 185, 65, 198, 163, 283, 287, ],
[98, 205, 214, 297, 127, 119, 100, 207, 187, 65, 77, 46, 372, 204, 220, 77, 235, 195, 337, 342, ],
[48, 99, 104, 144, 61, 58, 48, 100, 90, 31, 37, 23, 180, 99, 107, 37, 114, 94, 163, 165, ],
[53, 110, 115, 159, 68, 64, 54, 111, 100, 35, 41, 25, 199, 109, 118, 41, 126, 104, 180, 183, ],
[153, 317, 332, 460, 197, 185, 155, 320, 290, 100, 119, 72, 577, 316, 342, 120, 365, 302, 522, 530, ],
[128, 266, 278, 387, 165, 155, 130, 269, 243, 84, 100, 61, 484, 266, 287, 101, 306, 253, 439, 445, ],
[212, 440, 460, 639, 273, 256, 215, 444, 402, 139, 165, 100, 800, 439, 474, 166, 506, 418, 725, 735, ],
[26, 55, 58, 80, 34, 32, 27, 56, 50, 17, 21, 13, 100, 55, 59, 21, 63, 52, 91, 92, ],
[48, 100, 105, 146, 62, 58, 49, 101, 92, 32, 38, 23, 182, 100, 108, 38, 115, 95, 165, 167, ],
[45, 93, 97, 135, 58, 54, 45, 94, 85, 29, 35, 21, 169, 93, 100, 35, 107, 88, 153, 155, ],
[127, 264, 276, 384, 164, 154, 129, 267, 241, 83, 99, 60, 481, 264, 285, 100, 304, 251, 435, 441, ],
[42, 87, 91, 126, 54, 51, 42, 88, 79, 27, 33, 20, 158, 87, 94, 33, 100, 83, 143, 145, ],
[51, 105, 110, 153, 65, 61, 51, 106, 96, 33, 39, 24, 191, 105, 113, 40, 121, 100, 173, 176, ],
[29, 61, 63, 88, 38, 35, 30, 61, 55, 19, 23, 14, 110, 61, 65, 23, 70, 58, 100, 101, ],
[29, 60, 63, 87, 37, 35, 29, 60, 55, 19, 22, 14, 109, 60, 64, 23, 69, 57, 99, 100, ]],
[[100, 107, 213, 169, 80, 93, 156, 93, 140, 103, 117, 124, 355, 128, 122, 88, 92, 108, 69, 87, ],
[93, 100, 199, 158, 75, 86, 146, 87, 131, 96, 110, 116, 332, 120, 114, 82, 86, 101, 64, 81, ],
[47, 50, 100, 79, 38, 43, 73, 44, 66, 48, 55, 58, 167, 60, 57, 41, 43, 51, 32, 41, ],
[59, 63, 126, 100, 47, 55, 92, 55, 83, 61, 69, 73, 210, 76, 72, 52, 55, 64, 40, 51, ],
[125, 134, 266, 212, 100, 116, 195, 116, 175, 129, 147, 155, 444, 160, 153, 110, 115, 135, 86, 108, ],
[108, 116, 230, 183, 86, 100, 169, 100, 152, 111, 127, 134, 384, 139, 132, 95, 100, 117, 74, 94, ],
[64, 69, 136, 108, 51, 59, 100, 59, 90, 66, 75, 80, 228, 82, 78, 57, 59, 69, 44, 55, ],
[108, 115, 229, 182, 86, 100, 168, 100, 151, 111, 126, 134, 383, 138, 132, 95, 99, 117, 74, 93, ],
[71, 76, 152, 121, 57, 66, 111, 66, 100, 73, 84, 88, 253, 91, 87, 63, 66, 77, 49, 62, ],
[97, 104, 207, 164, 78, 90, 151, 90, 136, 100, 114, 120, 345, 124, 119, 86, 90, 105, 67, 84, ],
[85, 91, 182, 144, 68, 79, 133, 79, 120, 88, 100, 106, 303, 109, 104, 75, 79, 92, 58, 74, ],
[81, 86, 172, 136, 64, 75, 126, 75, 113, 83, 95, 100, 286, 103, 99, 71, 74, 87, 55, 70, ],
[28, 30, 60, 48, 23, 26, 44, 26, 39, 29, 33, 35, 100, 36, 34, 25, 26, 30, 19, 24, ],
[78, 83, 166, 132, 62, 72, 122, 72, 109, 80, 91, 97, 277, 100, 95, 69, 72, 84, 53, 68, ],
[82, 87, 174, 138, 65, 76, 128, 76, 115, 84, 96, 101, 290, 105, 100, 72, 75, 88, 56, 71, ],
[113, 121, 241, 192, 91, 105, 177, 105, 159, 117, 133, 141, 403, 145, 139, 100, 105, 123, 78, 98, ],
[108, 116, 231, 183, 87, 100, 169, 101, 152, 112, 127, 134, 385, 139, 133, 96, 100, 117, 74, 94, ],
[92, 99, 197, 156, 74, 85, 144, 86, 130, 95, 108, 115, 328, 118, 113, 81, 85, 100, 63, 80, ],
[146, 156, 311, 247, 117, 135, 228, 135, 205, 150, 171, 181, 518, 187, 179, 129, 135, 158, 100, 126, ],
[116, 124, 246, 196, 92, 107, 180, 107, 162, 119, 136, 143, 410, 148, 141, 102, 107, 125, 79, 100, ]],
[[100, 152, 123, 69, 50, 119, 473, 164, 728, 171, 206, 91, 236, 131, 327, 132, 116, 104, 157, 68, ],
[66, 100, 81, 46, 33, 78, 310, 107, 478, 112, 135, 60, 155, 86, 215, 87, 76, 68, 103, 45, ],
[81, 123, 100, 56, 41, 97, 383, 133, 590, 138, 167, 74, 191, 106, 265, 107, 94, 84, 127, 55, ],
[144, 220, 178, 100, 73, 172, 682, 236, 1050, 246, 297, 131, 341, 188, 472, 191, 167, 150, 226, 98, ],
[199, 303, 245, 138, 100, 237, 939, 325, 1447, 340, 410, 181, 470, 260, 650, 263, 230, 207, 312, 135, ],
[84, 128, 103, 58, 42, 100, 396, 137, 610, 143, 173, 76, 198, 109, 274, 111, 97, 87, 131, 57, ],
[21, 32, 26, 15, 11, 25, 100, 35, 154, 36, 44, 19, 50, 28, 69, 28, 24, 22, 33, 14, ],
[61, 93, 75, 42, 31, 73, 289, 100, 445, 105, 126, 56, 145, 80, 200, 81, 71, 64, 96, 42, ],
[14, 21, 17, 10, 7, 16, 65, 22, 100, 23, 28, 13, 32, 18, 45, 18, 16, 14, 22, 9, ],
[59, 89, 72, 41, 29, 70, 277, 96, 426, 100, 121, 53, 138, 76, 191, 77, 68, 61, 92, 40, ],
[49, 74, 60, 34, 24, 58, 229, 79, 353, 83, 100, 44, 115, 63, 159, 64, 56, 51, 76, 33, ],
[110, 167, 135, 76, 55, 131, 519, 179, 799, 187, 226, 100, 259, 143, 359, 145, 127, 114, 172, 75, ],
[42, 64, 52, 29, 21, 51, 200, 69, 308, 72, 87, 39, 100, 55, 138, 56, 49, 44, 66, 29, ],
[77, 117, 94, 53, 39, 91, 362, 125, 558, 131, 158, 70, 181, 100, 250, 101, 89, 80, 120, 52, ],
[31, 47, 38, 21, 15, 37, 145, 50, 223, 52, 63, 28, 72, 40, 100, 40, 35, 32, 48, 21, ],
[76, 115, 93, 52, 38, 90, 357, 124, 550, 129, 156, 69, 179, 99, 247, 100, 87, 79, 118, 51, ],
[86, 132, 107, 60, 44, 103, 409, 141, 630, 148, 178, 79, 204, 113, 283, 114, 100, 90, 136, 59, ],
[96, 146, 118, 67, 48, 115, 454, 157, 699, 164, 198, 87, 227, 125, 314, 127, 111, 100, 150, 65, ],
[64, 97, 79, 44, 32, 76, 302, 104, 464, 109, 131, 58, 151, 83, 209, 84, 74, 66, 100, 43, ],
[147, 224, 182, 102, 74, 176, 696, 241, 1072, 251, 303, 134, 348, 192, 481, 195, 170, 153, 231, 100, ]],
[[100, 300, 402, 152, 90, 719, 1516, 346, 553, 120, 257, 420, 88, 248, 527, 268, 197, 165, 376, 129, ],
[33, 100, 134, 51, 30, 240, 506, 115, 184, 40, 86, 140, 29, 83, 176, 90, 66, 55, 125, 43, ],
[25, 75, 100, 38, 22, 179, 377, 86, 137, 30, 64, 104, 22, 62, 131, 67, 49, 41, 94, 32, ],
[66, 197, 264, 100, 59, 473, 997, 227, 363, 79, 169, 276, 58, 163, 346, 176, 129, 109, 247, 85, ],
[111, 334, 448, 169, 100, 801, 1688, 385, 615, 134, 286, 467, 98, 276, 587, 299, 219, 184, 419, 144, ],
[14, 42, 56, 21, 12, 100, 211, 48, 77, 17, 36, 58, 12, 34, 73, 37, 27, 23, 52, 18, ],
[7, 20, 27, 10, 6, 47, 100, 23, 36, 8, 17, 28, 6, 16, 35, 18, 13, 11, 25, 9, ],
[29, 87, 116, 44, 26, 208, 438, 100, 160, 35, 74, 121, 25, 72, 152, 78, 57, 48, 109, 37, ],
[18, 54, 73, 28, 16, 130, 274, 63, 100, 22, 46, 76, 16, 45, 95, 49, 36, 30, 68, 23, ],
[83, 250, 335, 127, 75, 599, 1263, 288, 460, 100, 214, 349, 73, 206, 439, 224, 164, 138, 313, 108, ],
[39, 117, 157, 59, 35, 280, 591, 135, 216, 47, 100, 164, 34, 97, 206, 105, 77, 64, 147, 50, ],
[24, 71, 96, 36, 21, 171, 361, 82, 132, 29, 61, 100, 21, 59, 126, 64, 47, 39, 90, 31, ],
[113, 340, 456, 173, 102, 816, 1720, 392, 627, 136, 291, 476, 100, 281, 598, 304, 223, 187, 427, 146, ],
[40, 121, 162, 61, 36, 290, 612, 140, 223, 48, 104, 169, 36, 100, 213, 108, 79, 67, 152, 52, ],
[19, 57, 76, 29, 17, 136, 288, 66, 105, 23, 49, 80, 17, 47, 100, 51, 37, 31, 71, 25, ],
[37, 112, 150, 57, 33, 268, 565, 129, 206, 45, 96, 156, 33, 92, 196, 100, 73, 62, 140, 48, ],
[51, 153, 205, 77, 46, 366, 772, 176, 281, 61, 131, 213, 45, 126, 268, 137, 100, 84, 191, 66, ],
[60, 181, 243, 92, 54, 435, 917, 209, 334, 73, 155, 254, 53, 150, 319, 162, 119, 100, 228, 78, ],
[27, 80, 107, 40, 24, 191, 403, 92, 147, 32, 68, 112, 23, 66, 140, 71, 52, 44, 100, 34, ],
[77, 232, 311, 118, 70, 557, 1174, 268, 428, 93, 199, 325, 68, 192, 408, 208, 152, 128, 291, 100, ]],
[[100, 123, 661, 382, 30, 369, 652, 9, 554, 20, 36, 539, 512, 682, 868, 816, 481, 52, 338, 451, ],
[81, 100, 537, 310, 24, 299, 529, 7, 450, 16, 29, 437, 416, 554, 705, 663, 390, 43, 274, 366, ],
[15, 19, 100, 58, 5, 56, 99, 1, 84, 3, 5, 81, 77, 103, 131, 123, 73, 8, 51, 68, ],
[26, 32, 173, 100, 8, 97, 171, 2, 145, 5, 9, 141, 134, 179, 227, 214, 126, 14, 88, 118, ],
[334, 411, 2207, 1275, 100, 1231, 2176, 31, 1850, 68, 119, 1798, 1709, 2278, 2898, 2725, 1604, 175, 1127, 1507, ],
[27, 33, 179, 104, 8, 100, 177, 2, 150, 5, 10, 146, 139, 185, 236, 221, 130, 14, 92, 122, ],
[15, 19, 101, 59, 5, 57, 100, 1, 85, 3, 5, 83, 79, 105, 133, 125, 74, 8, 52, 69, ],
[1088, 1341, 7198, 4157, 326, 4013, 7096, 100, 6033, 220, 388, 5864, 5573, 7428, 9451, 8887, 5232, 571, 3675, 4913, ],
[18, 22, 119, 69, 5, 67, 118, 2, 100, 4, 6, 97, 92, 123, 157, 147, 87, 9, 61, 81, ],
[494, 609, 3267, 1887, 148, 1821, 3221, 45, 2738, 100, 176, 2661, 2529, 3371, 4289, 4033, 2374, 259, 1668, 2230, ],
[281, 346, 1857, 1072, 84, 1035, 1831, 26, 1556, 57, 100, 1513, 1438, 1916, 2438, 2292, 1350, 147, 948, 1267, ],
[19, 23, 123, 71, 6, 68, 121, 2, 103, 4, 7, 100, 95, 127, 161, 152, 89, 10, 63, 84, ],
[20, 24, 129, 75, 6, 72, 127, 2, 108, 4, 7, 105, 100, 133, 170, 159, 94, 10, 66, 88, ],
[15, 18, 97, 56, 4, 54, 96, 1, 81, 3, 5, 79, 75, 100, 127, 120, 70, 8, 49, 66, ],
[12, 14, 76, 44, 3, 42, 75, 1, 64, 2, 4, 62, 59, 79, 100, 94, 55, 6, 39, 52, ],
[12, 15, 81, 47, 4, 45, 80, 1, 68, 2, 4, 66, 63, 84, 106, 100, 59, 6, 41, 55, ],
[21, 26, 138, 79, 6, 77, 136, 2, 115, 4, 7, 112, 107, 142, 181, 170, 100, 11, 70, 94, ],
[191, 235, 1261, 728, 57, 703, 1243, 18, 1057, 39, 68, 1027, 976, 1301, 1655, 1557, 916, 100, 644, 861, ],
[30, 36, 196, 113, 9, 109, 193, 3, 164, 6, 11, 160, 152, 202, 257, 242, 142, 16, 100, 134, ],
[22, 27, 147, 85, 7, 82, 144, 2, 123, 4, 8, 119, 113, 151, 192, 181, 106, 12, 75, 100, ],
]
];

function run(peptfile::String, tcrfile::String)
    pepts = readlines(open(peptfile))
    tcrs = readlines(open(tcrfile))#readlines(open("./data/tcrs10000.txt"))
    #strengths = Array{Cint,1}(undef, size(pepts))
    #@time calc_binding_strengths(strengths, pepts, tcrs)
    @time calc_binding_strengths(pepts, 9, length(pepts), tcrs, length(tcrs))
end
#run("./data/uniquepeptides.txt", "./data/tcrs40000.txt")

#println(strengths)
#println(length(strengths))
end
