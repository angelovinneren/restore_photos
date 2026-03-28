using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;
using System.IO;

Console.WriteLine("Welcome to RestorePhotos!");

// Checking if we can get version via reflection as suggested
var emguVersion = typeof(CvInvoke).Assembly.GetName().Version;
Console.WriteLine($"Emgu.CV Assembly Version: {emguVersion}");

// Create a simple blank image
using Mat image = new Mat(480, 640, DepthType.Cv8U, 3);
image.SetTo(new MCvScalar(255, 0, 0)); // Blue background

// Draw some text
CvInvoke.PutText(
    image, 
    "Emgu.CV is Working!", 
    new Point(50, 240), 
    FontFace.HersheySimplex, 
    1.5, 
    new MCvScalar(255, 255, 255), 
    2);

// Save the image
string outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "test_output.png");
CvInvoke.Imwrite(outputPath, image);

Console.WriteLine($"Image created and saved to: {outputPath}");
Console.WriteLine("Press any key to exit.");
if (!Console.IsInputRedirected)
{
    Console.ReadKey();
}


// Cargar imagen
Mat img = CvInvoke.Imread("Chica.png", ImreadModes.ColorRgb);

// Resultado
Mat denoised = new Mat();

// DENOISE SEGURO
CvInvoke.FastNlMeansDenoisingColored(
    img,
    denoised,
    3,  // h luminance
    3,  // h color
    7,
    21
);

// Guardar resultado
CvInvoke.Imwrite("output_denoise.jpg", denoised);