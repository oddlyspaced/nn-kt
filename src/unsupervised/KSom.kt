package unsupervised

import kotlin.math.pow

class KSom {

    private val weights =
        arrayOf(
            arrayOf(0.15, 0.25, 0.85, 0.95),
            arrayOf(0.7, 0.8, 0.6, 0.4),
        )

    private val vectors = arrayOf(0.0, 1.0, 0.0, 1.0)
    private val alpha = 0.5


    fun calculate() {
        println("Calculate the euclidean dist")
        println("D(j)' = sigma i (w ij - x i) ^ 2")
        val dVals = arrayListOf<Double>()
        weights.forEachIndexed { index, wt ->
            println("D(${index + 1}) = sigma i=1 ${wt.size} (w ij - xi) ^ 2")
            var t = 0.0
            wt.forEachIndexed { index_in, vv ->
                t += (vv - vectors[index_in]).pow(2.0)
            }
            println(t)
            dVals.add(t)
        }

        val min = if (dVals[0] < dVals[1]) 0 else 1

        println()
        println("Since D(${min + 1}) is greater therefore winning cluster is ${min + 1}")
        println("Updating weights for J = ${min + 1}")

        weights[min].forEachIndexed { index, d ->
            weights[min][index] += alpha * (vectors[index] - d)
        }

        weights.forEach { w ->
            w.forEach {
                print("$it\t")
            }
            println()
        }

//        weights.forEach { w ->
//            w.forEach {
//                print("$it\t")
//            }
//            println()
//        }
    }
}