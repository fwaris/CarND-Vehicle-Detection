//script file for REPL evaluation of processing code
#load "SetEnv.fsx"
open OpenCVCommon
open OpenCvSharp
open CalibrateCamera
open FSharp.Charting
open System
open System.IO
open SetEnv 
open BaseTypes
open ImageProc
open Utils
open LaneFind
open LineFitting
open VideoProcessing

let v_fldr = @"D:\repodata\adv_lane_find"
let v_chlng2 = Path.Combine(v_fldr,"challenge_video.mp4")
let v_chlng1 = Path.Combine(v_fldr,"harder_challenge_video.mp4")
let v_prjct  = Path.Combine(v_fldr,"project_video.mp4")

let testHog() =
    let i_path = @"D:\repodata\adv_lane_find\imgs\img2.jpg"
    use img = Cv2.ImRead(i_path)
    //use img8 = new Mat()
    use gray = new Mat()
    //Cv2.CvtColor(!> img, !>img8, ColorConversionCodes.BGR2RGBA)
    Cv2.CvtColor(!> img, !>gray, ColorConversionCodes.BGR2GRAY)
    use hd = new HOGDescriptor()
    hd.WinSize <- img.Size()
    hd.BlockSize <- Size(16.,16.)
    hd.BlockStride <- Size(8.,8.)
    hd.CellSize <- Size(8.,8.)
    hd.Nbins <- 9
    hd.DerivAperture <- 0
    hd.WinSigma <- 1.
    hd.HistogramNormType <- HistogramNormType.L2Hys
    hd.L2HysThreshold <- 0.2
    hd.GammaCorrection <- false
    let _ = hd.CheckDetectorSize()
    let pts = hd.Compute(gray)
    let ptsM = pts |> Array.filter (fun x->x<0.f)
    hd.GetDescriptorSize()
    pts.Length / 9 / 8 / 8
    use m = ML.SVM.Create()
    //blocks 1280/8 - 1 = 159; 720/8 - 1 = 89
    

