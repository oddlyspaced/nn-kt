package unsupervised

import kotlin.math.pow

class Backpropagation {

    fun calculate() {
        val alpha = 0.25
        val inp = arrayOf(1.0, -1.0)
        val zWts =
            arrayOf(
                arrayOf(0.6, -0.1),
                arrayOf(-0.3, 0.4),
            )
        val zBias = arrayOf(0.3, 0.5)

        val yinWts = arrayOf(0.4, 0.1)
        val yinBias = -0.2

        println("Forward Pass\n")

        val zLayerRes = arrayListOf<Double>()

        for (i in zWts.indices) {
            val wt = zWts[i]
            println("Calculating net inp for z${i + 1} layer")
            var z = zBias[i] // 0.3

            for (j in inp.indices) {
                z += (inp[j] * wt[j])
            }

            printCustom(z)
            zLayerRes.add(z)
            println()
        }

        println("Apply activation function")
        val zLayerFuncRes = arrayListOf<Double>()

        zLayerRes.forEachIndexed { i: Int, v: Double ->
            print("Z${i+1} = f(Z${i+1}) = <fun> = ")
            zLayerFuncRes.add(activationFunc(v))
            printCustom(zLayerFuncRes[i])
            println()
        }

        println("\nCalculate net inp for op layer yin")
        var yin = yinBias

        zLayerFuncRes.forEachIndexed { index, v ->
            yin += (yinWts[index] * v)
        }

        println("yin:")
        printCustom(yin)
        println()

        println("Apply activation func:")
        println("y = f(yin) = <fun> = ")
        val fYin = activationFunc(yin)
        printCustom(fYin)

        // Backward pass

        println("Compute error portion ẟk")
        println("ẟk = (tk - yk) f'(yin)")

        println("f'(yin) = f(yin)[1 - f(yin)] = ")
        printCustom(fYin)
        print("[ 1 - ")
        printCustom(fYin)
        print("] = ")

        val fDashYIn = fYin * (1 - fYin)
        printCustom(fDashYIn)
        println()
        println()

        print("ẟ1 = [1 - ")
        printCustom(fYin)
        print("] ( ")
        printCustom(fDashYIn)
        print(") = ")
        val del1 = (1 - fYin) * fDashYIn
        printCustom(del1)

        println()
        println()
        println("Find change in wts between hidden and op layer")

        repeat(zLayerFuncRes.size) {
            print("ΔW${it+1} = αẟ1z${it+1} = <val> = ")
            printCustom((alpha * del1 * zLayerFuncRes[it]))
            println()
        }

        print("ΔW0 = αẟ1 = <val> = ")
        printCustom((alpha * del1))
        println()

        println("--")

        println("compute error portion ẟj b/w hidden and inp layer")
        println("ẟinj = ẟ1 wj 1 (only 1 op neuron)")

        val del1YinWtsRes = arrayListOf<Double>()

        yinWts.forEachIndexed { index, v ->
            print("ẟin${index + 1} = ẟ1w${index + 1}1 = <del 1 * wt> = ")
            del1YinWtsRes.add(del1 * v)
            printCustom(del1YinWtsRes[index])
            println()
        }

        val errorsRes = arrayListOf<Double>()
        println("Errors :")
        zLayerFuncRes.forEachIndexed { index, v ->
            println("Error ${index + 1}\nẟ${index + 1}= ẟin${index + 1} f'(zin ${index + 1})")
            println("f'(zin ${index + 1}) = f(zin ${index + 1})[1 - f(zin ${index + 1})]")
            printCustom(v)
            print("[ 1 - ")
            printCustom(v)
            print("] = ")
            val tempSave = (1 - v) * v
            printCustom(tempSave)
            println()
            print("ẟ${index + 1} = ")
            printCustom(del1YinWtsRes[index])
            print(" * ")
            printCustom(tempSave)
            print("= ")
            printCustom(tempSave * del1YinWtsRes[index])
            errorsRes.add(tempSave * del1YinWtsRes[index])
            println()
            println()
        }

        println("==================")

        println("Now find weight change bw hidden and input layer")
        errorsRes.forEachIndexed { index, v ->
            inp.forEachIndexed { index_inp, v_inp ->
                print("Δv${index_inp + 1}${index+1} = αẟ${index + 1}x${index_inp+1} = ")
                printCustom(alpha)
                printCustom(" * ")
                printCustom(v)
                printCustom(" * ")
                printCustom(v_inp)
                printCustom(" = ")
                val tt = alpha * v * v_inp
                printCustom(tt)
                println()
                println()
            }
        }
    }

    private fun activationFunc(z: Double): Double {
        return 1.0 / (1 + (Math.E.pow(-z)))
    }

    private fun printCustom(str: String) {
        print(str)
        repeat(10 - str.length) {
            print(" ")
        }
    }

    private fun printCustom(v: Double) {
        var ss = "$v"
        if (ss.length > 3) {
            ss = ss.substring(0, if (ss.length > 6) 6 else ss.length)
        }
        printCustom(ss)
    }

}