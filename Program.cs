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
        Console.WriteLine("🔥 RESTORE PRO RL + FACE SAFE");

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
        // 🔁 70+ COMBINACIONES PRO (CATEGORIZADAS)
        // =========================
        var configs = new List<(string cat, int h, int iter, int k, double sigma, double sharp, double clip)>
        {
            // 🔻 01_MUY_SUAVE (micro ajustes)
            ("01_MUY_SUAVE", 1, 3, 3, 0.3, 0.1, 0.8),
            ("01_MUY_SUAVE", 1, 5, 3, 0.5, 0.2, 1.0),
            ("01_MUY_SUAVE", 3, 3, 3, 0.4, 0.15, 1.0),
            ("01_MUY_SUAVE", 3, 5, 3, 0.6, 0.25, 1.2),
            ("01_MUY_SUAVE", 3, 8, 3, 0.7, 0.3, 1.5),

            // 🔸 02_SUAVE_MEDIO
            ("02_SUAVE_MEDIO", 3,10, 5, 1.0, 0.4, 2.0),
            ("02_SUAVE_MEDIO", 3,12, 5, 1.2, 0.5, 2.2),
            ("02_SUAVE_MEDIO", 5, 6, 3, 1.0, 0.3, 1.8),
            ("02_SUAVE_MEDIO", 5, 8, 5, 1.2, 0.5, 2.2),
            ("02_SUAVE_MEDIO", 5,10, 5, 1.5, 0.7, 2.5),
            ("02_SUAVE_MEDIO", 5,12, 5, 1.8, 0.8, 2.8),
            ("02_SUAVE_MEDIO", 5,15, 7, 2.0, 1.0, 3.2),
            ("02_SUAVE_MEDIO", 3,18, 5, 1.0, 0.6, 2.0),

            // 🔥 03_MEDIO
            ("03_MEDIO", 7, 6, 3, 0.8, 0.4, 1.5),
            ("03_MEDIO", 7, 8, 3, 1.0, 0.5, 2.0),
            ("03_MEDIO", 7,10, 5, 1.5, 0.8, 2.5),
            ("03_MEDIO", 7,12, 5, 1.8, 1.0, 3.0),
            ("03_MEDIO", 7,15, 7, 2.2, 1.2, 3.8),
            ("03_MEDIO", 7,18, 7, 2.5, 1.3, 4.5),
            ("03_MEDIO", 7,20, 7, 3.0, 1.5, 5.0),

            // 🔥🔥 04_FUERTE (restauración profunda)
            ("04_FUERTE", 9,10, 5, 2.0, 1.0, 3.0),
            ("04_FUERTE", 9,12, 7, 2.5, 1.2, 4.0),
            ("04_FUERTE", 9,15, 7, 3.0, 1.5, 5.0),
            ("04_FUERTE", 9,18, 9, 3.5, 1.7, 6.0),
            ("04_FUERTE", 9,20, 9, 4.0, 2.0, 7.0),
            ("04_FUERTE", 11,10, 5, 1.5, 0.8, 2.5),
            ("04_FUERTE", 11,12, 7, 2.0, 1.0, 3.5),

            // ⚡ 05_POTENTE (elimina grano pesado)
            ("05_POTENTE", 13,10, 5, 1.5, 0.7, 2.0),
            ("05_POTENTE", 13,12, 7, 2.0, 0.9, 3.0),
            ("05_POTENTE", 13,15, 7, 2.5, 1.2, 4.0),
            ("05_POTENTE", 13,18, 9, 3.0, 1.5, 5.0),
            ("05_POTENTE", 13,20, 9, 3.5, 1.8, 6.0),

            // 💀 06_AGRESIVO
            ("06_AGRESIVO", 15,15, 7, 3.0, 1.5, 5.0),
            ("06_AGRESIVO", 15,20, 9, 4.0, 2.0, 7.0),
            ("06_AGRESIVO", 15,25, 9, 5.0, 2.5, 8.0),
            ("06_AGRESIVO", 17,20, 9, 4.5, 2.2, 7.5),
            ("06_AGRESIVO", 17,25,11, 6.0, 3.0,10.0),

            // 💀💀 07_EXTREMO
            ("07_EXTREMO", 19,25,11, 7.0, 3.5,12.0),
            ("07_EXTREMO", 19,30,11, 8.0, 4.0,15.0),
            ("07_EXTREMO", 21,30,13, 9.0, 5.0,20.0),
            ("07_EXTREMO", 23,30,15,10.0, 6.0,25.0),
            ("07_EXTREMO", 25,35,17,12.0, 7.5,30.0),

            // 🔀 08_RANDOM (mezcla realista + caótica)
            ("08_RANDOM", 3,25, 3, 0.5, 2.0, 5.0),
            ("08_RANDOM", 5,30, 5, 1.0, 2.5, 6.0),
            ("08_RANDOM", 7,35, 7, 1.5, 3.0, 7.0),
            ("08_RANDOM", 9,40, 9, 2.0, 3.5, 8.0),
            ("08_RANDOM", 11,45,11,2.5, 4.0, 9.0),
            ("08_RANDOM", 5,15,15, 4.0, 3.0, 8.0),
            ("08_RANDOM", 7,20,17, 5.0, 3.5,10.0),
            ("08_RANDOM", 9,25,19, 6.0, 4.0,12.0),
            ("08_RANDOM", 11,30,21,7.0, 4.5,14.0),
            ("08_RANDOM", 13,35,23,8.0, 5.0,16.0),

            // 🔥🔥🔥 09_CAOS_CONTROLADO
            ("09_CAOS_CONTROLADO", 3,50, 3, 0.3, 3.0,10.0),
            ("09_CAOS_CONTROLADO", 5,50, 5, 0.8, 3.5,12.0),
            ("09_CAOS_CONTROLADO", 7,50, 7, 1.2, 4.0,14.0),
            ("09_CAOS_CONTROLADO", 9,50, 9, 2.0, 4.5,16.0),
            ("09_CAOS_CONTROLADO", 11,50,11,3.0, 5.0,18.0),

            // 🔻 10_EXTRA_BAJOS (no pierden detalle fino)
            ("10_EXTRA_BAJOS", 1,10, 3, 0.2, 0.05, 0.5),
            ("10_EXTRA_BAJOS", 1,15, 3, 0.3, 0.1, 0.7),
            ("10_EXTRA_BAJOS", 3,20, 3, 0.5, 0.2, 1.0),
            ("10_EXTRA_BAJOS", 3,25, 3, 0.7, 0.3, 1.5),
            ("10_EXTRA_BAJOS", 5,30, 3, 1.0, 0.4, 2.0)
        };


        int comboCount = configs.Count * 2; // (Original + 1 variante) -> No, triplica: (Original + 2 variantes)
        int totalCombos = configs.Count * 3;
        int currentIdx = 1;
        Random rnd = new Random();

        foreach (var cfg in configs)
        {
            for (int v = 0; v < 3; v++)
            {
                using Mat img = original.Clone();
                
                // Variaciones aleatorias para "triplicar" las posibilidades
                // v=0 (Original), v=1 (Variante A), v=2 (Variante B)
                double rv = v == 0 ? 0 : (rnd.NextDouble() * 0.4 + 0.8); // Mutaciones de 0.8x a 1.2x aproximadamente
                
                int h = v == 0 ? cfg.h : Math.Max(1, (int)(cfg.h * rv));
                int it = v == 0 ? cfg.iter : Math.Max(3, (int)(cfg.iter * (rnd.NextDouble() * 0.5 + 0.75)));
                double sh = v == 0 ? cfg.sharp : (cfg.sharp * (rnd.NextDouble() * 0.6 + 0.7));
                double cl = v == 0 ? cfg.clip : (cfg.clip * (rnd.NextDouble() * 0.4 + 0.8));

                // Crear carpeta de categoría si no existe
                string catDir = Path.Combine(rootFolder, cfg.cat);
                Directory.CreateDirectory(catDir);

                // =========================
                // 1. DENOISE
                // =========================
                using Mat denoise = new Mat();
                CvInvoke.FastNlMeansDenoisingColored(img, denoise, h, h, 7, 21);

                // =========================
                // 2. LAB (SEPARAR LUZ)
                // =========================
                using Mat lab = new Mat();
                CvInvoke.CvtColor(denoise, lab, ColorConversion.Bgr2Lab);

                using VectorOfMat ch = new VectorOfMat();
                CvInvoke.Split(lab, ch);

                Mat l = ch[0];
                Mat a = ch[1];
                Mat b = ch[2];

                // =========================
                // 💀 3. RL SOLO EN L
                // =========================
                using Mat l_rl = RichardsonLucy(l, it, cfg.k, cfg.sigma);

                // =========================
                // 🔍 4. SHARPEN CONTROLADO
                // =========================
                using Mat blur = new Mat();
                CvInvoke.GaussianBlur(l_rl, blur, new Size(0, 0), 1.2);

                using Mat sharp = new Mat();
                CvInvoke.AddWeighted(l_rl, 1 + sh, blur, -sh, 0, sharp);

                // =========================
                // 🌗 5. CLAHE
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
                // 🚀 6. HD
                // =========================
                using Mat hd = new Mat();
                CvInvoke.Resize(final, hd, new Size(final.Width * 2, final.Height * 2), 0, 0, Inter.Lanczos4);

                string fileName = $"img_{currentIdx}_h{h}_it{it}_s{sh:F1}_c{cl:F1}_v{v}_HD.jpg";
                string fullPath = Path.Combine(catDir, fileName);

                CvInvoke.Imwrite(fullPath, hd);
                Console.WriteLine($"[{currentIdx}/{totalCombos}] ✅ {cfg.cat} -> {fileName}");

                currentIdx++;
                l_rl?.Dispose();
            }
        }

        // =========================
        // 🧠 MODO ROSTRO SEGURO
        // =========================
        Console.WriteLine("\n🧠 PROCESANDO ROSTRO SEGURO...");

        string faceDir = Path.Combine(rootFolder, "06_ROSTROS");
        Directory.CreateDirectory(faceDir);

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

                // RL MÁS SUAVE (SEGURIDAD)
                Mat l_rl = RichardsonLucy(l, 6, 5, 1.2);

                using Mat blur = new Mat();
                CvInvoke.GaussianBlur(l_rl, blur, new Size(0, 0), 1.0);

                CvInvoke.AddWeighted(l_rl, 1.2, blur, -0.2, 0, l);

                CvInvoke.CLAHE(l, 2.0, new Size(8, 8), l);

                CvInvoke.Merge(ch, lab);
            }

            CvInvoke.CvtColor(lab, final, ColorConversion.Lab2Bgr);

            CvInvoke.Resize(final, hd, new Size(final.Width * 2, final.Height * 2), 0, 0, Inter.Lanczos4);

            string path = Path.Combine(faceDir, "Face_Final_HD.jpg");
            CvInvoke.Imwrite(path, hd);

            Console.WriteLine($"✅ ROSTRO FINAL: {path}");
        }

        Console.WriteLine("\n🔥 DONE");
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