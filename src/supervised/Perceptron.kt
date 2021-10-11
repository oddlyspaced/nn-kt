package supervised

/**
 * @param input x1 x2 ... xN
 * @param output t values for input
 * @param weights initial weights
 * @param bias initial bias
 * @param theta threshold
 * @param alpha learning rate
 */
class Perceptron(
    private val input: Array<Array<Double>>,
    private val output: Array<Double>,
    private val weights: Array<Double>,
    private var bias: Double,
    private val theta: Double,
    private val alpha: Double
) {

    // Input is double dimensional array
    // x1 x2 x3 .... xN t
    fun calculate(epochs: Int) {
        printHeader(input[0].size)
        repeat(epochs) {
            for (i in input.indices) {
                val row = input[i]
                val t = output[i]
                printRow(row, t)
                if (!calculateYIn(row, t)) {
                    updateWeights(row, t)
                }
                else {
                    print("Weights same")
                }
                println()
            }
            println("=== Epoch ${it + 1} Complete ===")
        }
    }

    /**
     * @param nCount Number of x in input
     */
    private fun printHeader(nCount: Int) {
        repeat(nCount) {
            printCustom("X${it + 1}")
        }
        printCustom("t")
        printCustom("y_in")
        printCustom("Y")
        repeat(nCount) {
            printCustom("ΔW${it + 1}")
        }
        printCustom("Δb")
        repeat(nCount) {
            printCustom("W${it + 1}")
        }
        printCustom("b")
        println()
    }

    private fun printRow(row: Array<Double>, t: Double) {
        row.forEach {
            printCustom(it.toString())
        }
        printCustom(t.toString())
    }

    // last is t
    private fun calculateYIn(row: Array<Double>, t: Double): Boolean {
        var res = 0.0
        for (i in row.indices) {
            res += row[i] * weights[i]
        }
        res += bias
        printCustom(res.toString())
        return calcY(res) == t
    }

    private fun calcY(yIn: Double): Double {
        val y = when {
            yIn > theta -> 1.0
            yIn >= -theta && yIn <= theta -> 0.0
            yIn < theta -> -1.0
            else -> -1.0
        }
        printCustom(y.toString())
        return y
    }

    private fun updateWeights(row: Array<Double>, t: Double) {
        for (i in row.indices) {
            val delta = alpha * t * row[i]
            printCustom(delta.toString())
            weights[i] = weights[i] + delta
        }
        updateBias(t)
        weights.forEach {
            printCustom(it.toString())
        }
        printCustom(bias.toString())
    }

    // Calculates delta bias and new bias also prints it
    private fun updateBias(t: Double) {
        val delta = alpha * t
        printCustom(delta.toString())
        bias += delta
    }

    // Prints in a custom length specific format
    private fun printCustom(str: String) {
        print(str)
        repeat(10 - str.length) {
            print(" ")
        }
    }


}