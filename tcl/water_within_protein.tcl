#execute: vmd -dispdev text -e water_within_protein.tcl -args CODE FILELIST PARAMETERS> water_count.out
# e.g: FOLDER_NAME = 6b73B
# e.g: PARAMETERS = 11 9 15 13 12 18 5

set code [lindex $argv 0]

mol new ../../$code/results/step5_assembly.xplor_ext.psf type psf
mol off top
set first_frame 0
set last_frame -1

set files [lindex $argv 1]
set filelist [split [lindex $files 0]]
set nf [llength $filelist]

set x [lindex $argv 2]
set y [lindex $argv 3]
set z [lindex $argv 4]
set _x [lindex $argv 5]
set _y [lindex $argv 6]
set _z [lindex $argv 7]
set distance [lindex $argv 8]

set selection "(water within $distance of protein) and (x<$x and y<$y and z<$z and -x<$_x and -y<$_y and -z<$_z) and oxygen"

for { set i 0 } { $i <= $nf } { incr i } {
    set file_name [lindex $filelist $i]
    puts file_name
    set crnt_file [glob ../$file_name-pbc.dcd]
    puts crnt_file
    
	animate read dcd $crnt_file beg 0 end -1 waitfor all

	set num_steps [molinfo top get numframes]
        
	#set out_file $crnt_file-warter_count.txt
    set out_file $crnt_file-warter_count.txt
    
	set fid [open $out_file w]

	for {set frame 0} {$frame < $num_steps} {incr frame} {
        puts "Frame: $frame"
        set a [atomselect top $selection frame $frame]
        set num [$a num]
        puts $fid "$frame $num"
        $a delete
	}
	#Close written files
	close $fid

	#set first_frame [expr {$first_frame + $num_steps}]
    animate delete all
}

puts "Done"
quit
