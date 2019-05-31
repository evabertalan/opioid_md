#execute: vmd -dispdev text -e water_near_residue.tcl -args CODE FILELIST SOD_RESID sod_x sod_y sod_z > sodium_coordinate_log.out
# e.g: CODE = 6b73B
# e.g: FILELIST = 
# e.g: SOD_RESID = 50
# e.g: sod_x = -4.959457
# e.g: sod_y = -6.712894
# e.g: sod_z = -3.926964

set code [lindex $argv 0]
set files [lindex $argv 1]
set filelist [split [lindex $files 0]]
set resid [lindex $argv 2]
set sod_x [lindex $argv 3]
set sod_y [lindex $argv 4]
set sod_z [lindex $argv 5]

set nf [llength $filelist]
set selection "resname SOD and resid $resid"

mol new ../../$code/results/step5_assembly.xplor_ext.psf type psf
mol off top
set first_frame 0
set last_frame -1

for { set i 0 } { $i <= $nf } { incr i } {
    puts [lindex $filelist $i]
    set file_name [lindex $filelist $i]
	set crnt_file [glob ../$file_name-pbc.dcd]

	animate read dcd $crnt_file beg 0 end -1 waitfor all

	set num_steps [molinfo top get numframes]
        
	#set out_file $crnt_file-warter_count.txt
    set out_file ../$file_name-sodium_coord.txt
    
	set fid [open $out_file w]

	for {set frame 0} {$frame < $num_steps} {incr frame} {
        puts "Frame: $frame"
        set a [atomselect top $selection frame $frame]
        
        set sod_center [measure center $a]
        set dest_point [list $sod_x $sod_y $sod_z ]
        
        puts $sod_center
        puts $dest_point
        
        set dist [vecdist $sod_center $dest_point]
        
        puts $fid "$frame $sod_center $dist"
        $a delete
	}
	#Close written files
	close $fid

	#set first_frame [expr {$first_frame + $num_steps}]
    animate delete all
}


puts "Done"
quit