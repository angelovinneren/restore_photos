using System;
using System.IO;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("🔥 RESTORE PRO v2 (COLOR + SAFE RL + DETAIL)");

        string input = args.Length > 0 ? args[0] : "foto.jpg";

        if (!File.Exists(input))
        {
            Console.WriteLine("❌ Imagen no encontrada");
            return;
        }

        Mat original = CvInvoke.Imread(input, ImreadModes.AnyColor);

        string rootFolder = Path.GetFileNameWithoutExtension(input);
        Directory.CreateDirectory(rootFolder);

        // =========================
        // 🎨 0. COLOR FIX (ANTI AMARILLO)
        // =========================
        using (Mat balanced = new Mat())
        {
            CvInvoke.CvtColor(original, balanced, ColorConversion.Bgr2Lab);

            using (VectorOfMat ch = new VectorOfMat())
            {
                CvInvoke.Split(balanced, ch);

                // Ajustes suaves (puedes variar -3 a -10)
                CvInvoke.Add(ch[1], new ScalarArray(-5), ch[1]);
                CvInvoke.Add(ch[2], new ScalarArray(-5), ch[2]);

                CvInvoke.Merge(ch, balanced);
            }

            CvInvoke.CvtColor(balanced, original, ColorConversion.Lab2Bgr);
        }

        // =========================
        // 🌗 1. CONTRASTE GLOBAL
        // =========================
        using (Mat labGlobal = new Mat())
        {
            CvInvoke.CvtColor(original, labGlobal, ColorConversion.Bgr2Lab);

            using (VectorOfMat ch = new VectorOfMat())
            {
                CvInvoke.Split(labGlobal, ch);

                CvInvoke.CLAHE(ch[0], 2.5, new Size(8, 8), ch[0]);

                CvInvoke.Merge(ch, labGlobal);
            }

            CvInvoke.CvtColor(labGlobal, original, ColorConversion.Lab2Bgr);
        }

        // =========================
        // 🔁 CONFIGURACIONES SEGURAS (CATEGORIZADAS)
        // =========================
        var configs = new List<(string cat, int h, int iter, int k, double sigma, double sharp, double clip)>
        {
            ("01_SUAVE", 3, 6, 3, 1.0, 0.4, 2.0),
            ("01_SUAVE", 5, 8, 3, 1.2, 0.6, 2.5),
            ("02_MEDIO", 5, 10, 5, 1.5, 0.8, 3.0),
            ("02_MEDIO", 7, 8, 3, 1.0, 0.5, 2.0),
            ("02_MEDIO", 7, 10, 5, 1.5, 0.7, 2.5)
        };

        int totalCombos = configs.Count * 3;
        int currentIdx = 1;
        Random rnd = new Random();

        foreach (var cfg in configs)
        {
            for (int v = 0; v < 3; v++)
            {
                using Mat img = original.Clone();

                // Variaciones aleatorias (Original + 2 variantes)
                double rv = v == 0 ? 0 : (rnd.NextDouble() * 0.4 + 0.8);
                int h = v == 0 ? cfg.h : Math.Max(1, (int)(cfg.h * rv));
                int it = v == 0 ? cfg.iter : Math.Max(3, (int)(cfg.iter * (rnd.NextDouble() * 0.5 + 0.75)));
                double sh = v == 0 ? cfg.sharp : (cfg.sharp * (rnd.NextDouble() * 0.6 + 0.7));
                double cl = v == 0 ? cfg.clip : (cfg.clip * (rnd.NextDouble() * 0.4 + 0.8));

                // Crear carpeta de categoría si no existe
                string catDir = Path.Combine(rootFolder, cfg.cat);
                Directory.CreateDirectory(catDir);

                // =========================
                // 2. DENOISE
                // =========================
                using Mat denoise = new Mat();
                CvInvoke.FastNlMeansDenoisingColored(img, denoise, h, h, 7, 21);

                // =========================
                // 3. LAB
                // =========================
                using Mat lab = new Mat();
                CvInvoke.CvtColor(denoise, lab, ColorConversion.Bgr2Lab);

                using VectorOfMat ch = new VectorOfMat();
                CvInvoke.Split(lab, ch);

                Mat l = ch[0];
                Mat a = ch[1];
                Mat b = ch[2];

                // =========================
                // 💀 RL SEGURO
                // =========================
                int safeIter = Math.Min(it, 12);
                double safeSigma = Math.Min(cfg.sigma, 2.5);

                using Mat l_rl = RichardsonLucy(l, safeIter, cfg.k, safeSigma);

                // =========================
                // 🔍 SHARP CONTROLADO
                // =========================
                using Mat blur = new Mat();
                CvInvoke.GaussianBlur(l_rl, blur, new Size(0, 0), 1.2);

                using Mat sharp = new Mat();
                double safeSharp = Math.Min(sh, 1.5);
                CvInvoke.AddWeighted(l_rl, 1 + safeSharp, blur, -safeSharp, 0, sharp);

                // =========================
                // 🧬 MICRO DETALLE (LAPLACIANO)
                // =========================
                using Mat detail = new Mat();
                CvInvoke.Laplacian(sharp, detail, DepthType.Cv16S, 3);

                using Mat absDetail = new Mat();
                CvInvoke.ConvertScaleAbs(detail, absDetail, 1.0, 0.0);

                CvInvoke.AddWeighted(sharp, 1.0, absDetail, 0.3, 0, sharp);

                // =========================
                // 🌗 CLAHE FINAL
                // =========================
                CvInvoke.CLAHE(sharp, cl, new Size(8, 8), l);

                using VectorOfMat merge = new VectorOfMat();
                merge.Push(l);
                merge.Push(a);
                merge.Push(b);

                using Mat merged = new Mat();
                CvInvoke.Merge(merge, merged);

                using Mat final = new Mat();
                CvInvoke.CvtColor(merged, final, ColorConversion.Lab2Bgr);

                // =========================
                // 🚀 HD
                // =========================
                using Mat hd = new Mat();
                CvInvoke.Resize(final, hd, new Size(final.Width * 2, final.Height * 2), 0, 0, Inter.Lanczos4);

                string fileName = $"img_{currentIdx}_h{h}_it{it}_v{v}_HD.jpg";
                string fullPath = Path.Combine(catDir, fileName);

                CvInvoke.Imwrite(fullPath, hd);
                Console.WriteLine($"[{currentIdx}/{totalCombos}] ✅ {cfg.cat} -> {fileName}");
                
                currentIdx++;
            }
        }

        // =========================
        // 🧠 FACE SAFE (ULTRA SUAVE)
        // =========================
        Console.WriteLine("🧠 Procesando rostro seguro...");

        using (Mat denoise = new Mat())
        using (Mat lab = new Mat())
        using (Mat final = new Mat())
        using (Mat hd = new Mat())
        {
            CvInvoke.FastNlMeansDenoisingColored(original, denoise, 3, 3, 7, 21);

            CvInvoke.CvtColor(denoise, lab, ColorConversion.Bgr2Lab);

            using (VectorOfMat ch = new VectorOfMat())
            {
                CvInvoke.Split(lab, ch);

                Mat l = ch[0];

                Mat l_rl = RichardsonLucy(l, 4, 3, 0.8);

                using Mat blur = new Mat();
                CvInvoke.GaussianBlur(l_rl, blur, new Size(0, 0), 1.0);

                CvInvoke.AddWeighted(l_rl, 1.1, blur, -0.1, 0, l);

                CvInvoke.CLAHE(l, 2.0, new Size(8, 8), l);

                CvInvoke.Merge(ch, lab);
            }

            CvInvoke.CvtColor(lab, final, ColorConversion.Lab2Bgr);

            CvInvoke.Resize(final, hd, new Size(final.Width * 2, final.Height * 2), 0, 0, Inter.Lanczos4);

            string facePath = Path.Combine(rootFolder, "Face_SAFE.jpg");
            CvInvoke.Imwrite(facePath, hd);

            Console.WriteLine($"✅ {facePath}");
        }

        Console.WriteLine("🔥 DONE");
        Console.ReadKey();
    }

    // =========================
    // 💀 RICHARDSON-LUCY
    // =========================
    static Mat RichardsonLucy(Mat input, int iterations, int kernelSize, double sigma)
    {
        Mat img = new Mat();
        input.ConvertTo(img, DepthType.Cv32F, 1.0 / 255.0);

        Mat psf = CvInvoke.GetGaussianKernel(kernelSize, sigma, DepthType.Cv32F);
        Mat psf2D = new Mat();
        CvInvoke.MulTransposed(psf, psf2D, false);
        CvInvoke.Normalize(psf2D, psf2D, 1.0, 0, NormType.L1);

        Mat psfFlip = new Mat();
        CvInvoke.Flip(psf2D, psfFlip, FlipType.Both);

        Mat estimate = img.Clone();
        Mat temp = new Mat(img.Size, DepthType.Cv32F, 1);
        Mat relative = new Mat(img.Size, DepthType.Cv32F, 1);

        for (int i = 0; i < iterations; i++)
        {
            CvInvoke.Filter2D(estimate, temp, psf2D, new Point(-1, -1));
            CvInvoke.Max(temp, new ScalarArray(1e-7), temp);

            CvInvoke.Divide(img, temp, relative);

            CvInvoke.Filter2D(relative, temp, psfFlip, new Point(-1, -1));
            CvInvoke.Multiply(estimate, temp, estimate);
        }

        Mat result = new Mat();
        estimate.ConvertTo(result, DepthType.Cv8U, 255.0);

        return result;
    }
}