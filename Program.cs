using System;
using System.IO;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.Reflection;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("🔥 RESTORE PRO v4 (INVISIBLE BLENDING + COLOR REVIVE)");

        if (args.Length >= 2)
        {
            Console.WriteLine("✂️ MODO CIRUJANO: Fusionando con Blending Pro...");
            StitchFaces(args[0], args[1]);
            return;
        }

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
        // 🎨 COLOR REVIVE (GRAY WORLD + VIVID)
        // =========================
        using (Mat labFix = new Mat())
        {
            CvInvoke.CvtColor(original, labFix, ColorConversion.Bgr2Lab);
            using (VectorOfMat ch = new VectorOfMat())
            {
                CvInvoke.Split(labFix, ch);

                MCvScalar meanA = CvInvoke.Mean(ch[1]);
                MCvScalar meanB = CvInvoke.Mean(ch[2]);
                
                CvInvoke.Add(ch[1], new ScalarArray(128 - meanA.V0), ch[1]);
                CvInvoke.Add(ch[2], new ScalarArray(128 - meanB.V0), ch[2]);

                CvInvoke.CLAHE(ch[0], 2.0, new Size(8, 8), ch[0]);

                CvInvoke.Merge(ch, labFix);
            }
            CvInvoke.CvtColor(labFix, original, ColorConversion.Lab2Bgr);
        }

        // =========================
        // 🔁 BIBLIOTECA MASIVA (RESTAURACION)
        // =========================
        var baseConfigs = new List<(string cat, int h, int iter, int k, double sigma, double sharp, double clip, double sat)>
        {
            ("01_NATURAL", 1, 3, 3, 0.3, 0.1, 1.0, 1.0),
            ("02_EQUILIBRADO", 5,10, 5, 1.5, 0.7, 2.5, 1.3),
            ("03_DETALLE", 5,15, 3, 0.8, 1.5, 4.0, 1.4),
            ("04_ANTIGUA_PRO", 9,10, 7, 1.4, 1.2, 3.5, 1.8),
            ("05_VARIACIONES", 7,30, 5, 1.2, 2.5, 7.0, 2.0)
        };

        Random rnd = new Random();

        foreach (var cfg in baseConfigs)
        {
            for (int v = 0; v < 3; v++)
            {
                using Mat img = original.Clone();

                double factor = v == 0 ? 1.0 : (rnd.NextDouble() * 0.4 + 0.8);
                int h = (int)(cfg.h * factor);
                int nt = (int)(cfg.iter * factor);
                double sh = cfg.sharp * factor;
                double cl = cfg.clip * factor;
                double sat = cfg.sat * factor;

                string catDir = Path.Combine(rootFolder, cfg.cat);
                Directory.CreateDirectory(catDir);

                using  Mat denoise = new Mat();
                CvInvoke.FastNlMeansDenoisingColored(img, denoise, h, h, 7, 21);

                using Mat lab = new Mat();
                CvInvoke.CvtColor(denoise, lab, ColorConversion.Bgr2Lab);
                using (VectorOfMat ch = new VectorOfMat())
                {
                    CvInvoke.Split(lab, ch);
                    if (sat > 1.0)
                    {
                        ch[1].ConvertTo(ch[1], DepthType.Cv32F);
                        ch[2].ConvertTo(ch[2], DepthType.Cv32F);
                        CvInvoke.Subtract(ch[1], new ScalarArray(128), ch[1]);
                        CvInvoke.Subtract(ch[2], new ScalarArray(128), ch[2]);
                        CvInvoke.Multiply(ch[1], new ScalarArray(sat), ch[1]);
                        CvInvoke.Multiply(ch[2], new ScalarArray(sat), ch[2]);
                        CvInvoke.Add(ch[1], new ScalarArray(128), ch[1]);
                        CvInvoke.Add(ch[2], new ScalarArray(128), ch[2]);
                        ch[1].ConvertTo(ch[1], DepthType.Cv8U);
                        ch[2].ConvertTo(ch[2], DepthType.Cv8U);
                    }
                    using (Mat l_rl = RichardsonLucy(ch[0], Math.Min(nt, 20), cfg.k, cfg.sigma))
                    {
                        using (Mat blur = new Mat())
                        {
                            CvInvoke.GaussianBlur(l_rl, blur, new Size(0, 0), 1.2);
                            CvInvoke.AddWeighted(l_rl, 1 + sh, blur, -sh, 0, ch[0]);
                            CvInvoke.CLAHE(ch[0], cl, new Size(8, 8), ch[0]);
                        }
                    }
                    CvInvoke.Merge(ch, lab);
                }

                using Mat final = new Mat();
                CvInvoke.CvtColor(lab, final, ColorConversion.Lab2Bgr);
                using Mat hd = new Mat();
                CvInvoke.Resize(final, hd, new Size(final.Width * 2, final.Height * 2), 0, 0, Inter.Lanczos4);

                string fname = $"img_v{v}_sat{sat:F1}.jpg";
                CvInvoke.Imwrite(Path.Combine(catDir, fname), hd);
            }
        }
    }

    // =========================
    // ✂️ MODO CIRUJANO PRO (FUSION INVISIBLE)
    // =========================
    static void StitchFaces(string pathO, string pathB)
    {
        string cascadePath = "haarcascade_frontalface_default.xml";
        if (!File.Exists(cascadePath))
        {
            Console.WriteLine("❌ ERROR: No se encuentra haarcascade_frontalface_default.xml");
            return;
        }

        using Mat imgO = CvInvoke.Imread(pathO, ImreadModes.AnyColor);
        using Mat imgB = CvInvoke.Imread(pathB, ImreadModes.AnyColor);

        if (imgO.IsEmpty || imgB.IsEmpty) return;

        using CascadeClassifier detector = new CascadeClassifier(cascadePath);
        Rectangle[] facesO = detector.DetectMultiScale(imgO, 1.1, 5);
        Rectangle[] facesB = detector.DetectMultiScale(imgB, 1.1, 5);

        Array.Sort(facesO, (a, b) => a.X.CompareTo(b.X));
        Array.Sort(facesB, (a, b) => a.X.CompareTo(b.X));

        using Mat composite = imgB.Clone();
        int matched = Math.Min(facesO.Length, facesB.Length);

        for (int i = 0; i < matched; i++)
        {
            Rectangle rO = facesO[i];
            Rectangle rB = facesB[i];

            // 1. Mejorar Brillo de la Cara O para que coincida con B
            using Mat faceO = new Mat(imgO, rO);
            using (Mat faceB_Region = new Mat(imgB, rB))
            {
                MatchBrightness(faceO, faceB_Region);
            }

            using Mat faceOResized = new Mat();
            CvInvoke.Resize(faceO, faceOResized, rB.Size, 0, 0, Inter.Lanczos4);

            // 2. Máscara de Degradado Radical (Evita el efecto mascarilla)
            using Mat mask = new Mat(rB.Size, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(0));
            // Ovalo central
            CvInvoke.Ellipse(mask, new Point(rB.Width / 2, rB.Height / 2), new Size(rB.Width / 3, (int)(rB.Height * 0.45)), 0, 0, 360, new MCvScalar(255), -1);
            // Suavizado extremo de la máscara
            int blurSize = (int)(rB.Width * 0.25); 
            if (blurSize % 2 == 0) blurSize++;
            CvInvoke.GaussianBlur(mask, mask, new Size(blurSize, blurSize), 0);

            // 3. Clonación Inconsútil con Reflexión
            Point center = new Point(rB.X + rB.Width / 2, rB.Y + rB.Height / 2);
            try
            {
                var methods = typeof(CvInvoke).GetMethods();
                var scMethod = Array.Find(methods, m => m.Name == "SeamlessClone" && m.GetParameters().Length == 6);
                if (scMethod != null)
                {
                    Type flagType = scMethod.GetParameters()[5].ParameterType;
                    object flagValue = Enum.ToObject(flagType, 1); // NormalClone
                    scMethod.Invoke(null, new object[] { faceOResized, composite, mask, center, composite, flagValue });
                    
                    // 4. Pulido Post-Fusión (Skin Paint)
                    ApplyBilateralPolish(composite, rB);
                    Console.WriteLine($"✨ Rostro {i + 1} fusionado con éxito.");
                }
            }
            catch { }
        }

        CvInvoke.Imwrite("FUSION_FINAL_PRO.jpg", composite);
        Console.WriteLine("\n🚀 ¡FUSION PRO COMPLETADA! Imagen: FUSION_FINAL_PRO.jpg");
    }

    static void MatchBrightness(Mat src, Mat target)
    {
        using Mat labSrc = new Mat();
        using Mat labTgt = new Mat();
        CvInvoke.CvtColor(src, labSrc, ColorConversion.Bgr2Lab);
        CvInvoke.CvtColor(target, labTgt, ColorConversion.Bgr2Lab);

        using (VectorOfMat chSrc = new VectorOfMat())
        using (VectorOfMat chTgt = new VectorOfMat())
        {
            CvInvoke.Split(labSrc, chSrc);
            CvInvoke.Split(labTgt, chTgt);

            double meanSrc = CvInvoke.Mean(chSrc[0]).V0;
            double meanTgt = CvInvoke.Mean(chTgt[0]).V0;

            CvInvoke.Add(chSrc[0], new ScalarArray(meanTgt - meanSrc), chSrc[0]);
            CvInvoke.Merge(chSrc, labSrc);
        }
        CvInvoke.CvtColor(labSrc, src, ColorConversion.Lab2Bgr);
    }

    static void ApplyBilateralPolish(Mat img, Rectangle rect)
    {
        // Pequeño margen para cubrir la transición
        Rectangle expand = new Rectangle(rect.X - 10, rect.Y - 10, rect.Width + 20, rect.Height + 20);
        expand.Intersect(new Rectangle(0, 0, img.Width, img.Height));

        using Mat roi = new Mat(img, expand);
        using Mat polished = new Mat();
        CvInvoke.BilateralFilter(roi, polished, 9, 75, 75);
        
        // Mezclar suavemente el ROI pulido
        CvInvoke.AddWeighted(roi, 0.4, polished, 0.6, 0, roi);
    }

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
        using Mat temp = new Mat();
        using Mat relative = new Mat();

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
