package supervised

/**
 * @param input x1 x2 ... xN
 * @param output t values for input
 * @param weights initial weights
 * @param bias initial bias
 * @param alpha learning rate
 */
class Adaline(
    private val input: Array<Array<Double>>,
    private val output: Array<Double>,
    private val weights: Array<Double>,
    private var bias: Double,
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
                val yIn = calculateYIn(row, t)
                val diffYIn = calcTDiffYIn(t, yIn)
                printCustom(diffYIn)
                updateWeights(row, t, yIn)
                sqSum += (diffYIn * diffYIn)
                printCustom(diffYIn * diffYIn)
                println()
            }
            println("Mean Square Sum: $sqSum")
            println("=== Epoch ${it + 1} Complete ===")
            sqSum = 0.0
        }
    }

    private var sqSum = 0.0

    /**
     * @param nCount Number of x in input
     */
    private fun printHeader(nCount: Int) {
        repeat(nCount) {
            printCustom("X${it + 1}")
        }
        arrayOf("t", "y_in", "(t-y_in)").forEach {
            printCustom(it)
        }

        repeat(nCount) {
            printCustom("ΔW${it + 1}")
        }

        printCustom("Δb")

        repeat(nCount) {
            printCustom("W${it + 1}")
        }
        printCustom("b")
        printCustom("(t-y_in)^2")
        println()
    }

    private fun printRow(row: Array<Double>, t: Double) {
        row.forEach {
            printCustom(it)
        }
        printCustom(t)
    }

    private fun calculateYIn(row: Array<Double>, t: Double): Double {
        var res = 0.0
        for (i in row.indices) {
            res += row[i] * weights[i]
        }
        res += bias
        printCustom(res)
        return res
    }

    private fun calcTDiffYIn(t: Double, yIn: Double): Double {
        return t - yIn
    }

    private fun updateWeights(row: Array<Double>, t: Double, yIn: Double) {
        for (i in row.indices) {
            val delta = alpha * calcTDiffYIn(t, yIn) * row[i]
            printCustom(delta)
            weights[i] = weights[i] + delta
        }
        updateBias(t, yIn)
        weights.forEach {
            printCustom(it)
        }
        printCustom(bias)
    }

    // Calculates delta bias and new bias also prints it
    private fun updateBias(t: Double, yIn: Double) {
        val delta = alpha * calcTDiffYIn(t, yIn)
        printCustom(delta)
        bias += delta
    }

    // Prints in a custom length specific format
    private fun printCustom(str: String) {
        print(str)
        repeat(10 - str.length) {
            print(" ")
        }
    }

    private fun printCustom(v: Double) {
        var ss = v.toString()
        if (ss.length > 3) {
            ss = ss.substring(0, if (ss.length > 5) 5 else ss.length)
        }
        printCustom(ss)
    }


}